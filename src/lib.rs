//! A rusty wrapper for Linux's epoll interface that is easy to use and hard
//! to misuse.
//!
//! Create a new epoll instance with [`Epoll::new`]. Add any struct
//! that implements the [`OwnedRawFd`] trait with [`Epoll::add`].
//! epoll::add returns a [`Token`] that takes ownership of the added file.
//! ```no_run
//! use epoll_rs::{Epoll, Opts};
//! # fn main() -> std::io::Result<()> {
//! let mut epoll = Epoll::new()?;
//! # let file = std::fs::File::open("")?;
//! let token = epoll.add(file, Opts::IN)?;
//! # Ok(())
//! # }
//! ```
//!
//! Tokens returned from one epoll instance cannot be used with another instance.
//! Doing so will cause a panic in debug mode and undefined behavior in release mode.
//! ```no_run
//! use epoll_rs::{Epoll, Opts};
//! # fn main() -> std::io::Result<()> {
//! let mut epoll1 = Epoll::new()?;
//! let mut epoll2 = Epoll::new()?;
//! # let file = std::fs::File::open("")?;
//! let token1 = epoll1.add(file, Opts::IN)?;
//! let res = epoll2.remove(token1); // <- undefined behavior in release mode
//! # Ok(())
//! # }
//! ```

use bitflags::bitflags;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::os::unix::{self, io::{AsRawFd, FromRawFd, IntoRawFd, RawFd}};
use std::{convert::TryInto, io, net, time::Duration};
#[cfg(not(debug_assertions))]
use std::marker::PhantomData;


/// Opaque type used to refer to single files registered with an epoll instance
///
/// In debug mode it has extra fields to ensure you're only using it with the
/// epoll instance it came from, but in release mode these fields are stripped
/// out.
#[derive(Debug)] //TODO: consider other derives (eq? hash?)
pub struct Token<'a, F: OwnedRawFd> {
    file: F,
    #[cfg(not(debug_assertions))]
    phantom: PhantomData<&'a Epoll>,
    #[cfg(debug_assertions)]
    epoll: &'a Epoll,
    #[cfg(debug_assertions)]
    epoll_fd: RawFd,
}

impl<'a, F: OwnedRawFd> Token<'a, F> {
    #[cfg(debug_assertions)]
    fn new(file: F, epoll: &'a Epoll) -> Self {
        Token {
            epoll_fd: epoll.epoll_fd,
            file,
            epoll,
        }
    }
    #[cfg(not(debug_assertions))]
    fn new(file: F) -> Self {
        Token {
            file,
            phantom: PhantomData,
        }
    }

    /// Consumes this token and returns the contained file
    ///
    /// This does not remove the file from any epoll instances it has been
    /// added to.
    pub fn into_file(self) -> F {
        self.file
    }

    /// Gives an immutable reference to the contained file
    pub fn file(&self) -> &F {
        &self.file
    }

    /// Equivalent to calling `self.file().as_raw_fd()`, only shorter
    ///
    /// Don't close the returned RawFd or create a `File` from it
    pub fn fd(&self) -> RawFd {
        self.file().as_raw_fd()
    }

    /// Gives a mutable reference to the contained file
    pub fn file_mut(&mut self) -> &mut F {
        &mut self.file
    }
}

bitflags! {
    /// Options used in [adding](Epoll::add) a file or
    /// [modifying](Epoll::modify) a previously added file
    ///
    /// Bitwise or (`|`) these together to combine multiple options
    pub struct Opts: libc::c_uint {
        /// Available for reads
        const IN = libc::EPOLLIN as libc::c_uint;
        /// Available for writes
        const OUT = libc::EPOLLOUT as libc::c_uint;
        /// Socket connection closed
        const RDHUP = libc::EPOLLRDHUP as libc::c_uint;
        /// Exceptional condition (see man 3 poll)
        const PRI = libc::EPOLLPRI as libc::c_uint;
        const ERR = libc::EPOLLERR as libc::c_uint;
        /// Hangup. If you register for another event type, this is automatically enabled
        const HUP = libc::EPOLLHUP as libc::c_uint;
        /// Use edge-triggered notifications
        const ET = libc::EPOLLET as libc::c_uint;
        const ONESHOT = libc::EPOLLONESHOT as libc::c_uint;
        const WAKEUP = libc::EPOLLWAKEUP as libc::c_uint;
        /// Deliver on only one epoll fd (see man epoll)
        const EXCLUSIVE = libc::EPOLLEXCLUSIVE as libc::c_uint;
    }
}

/// if condition is true, return errno
macro_rules! then_errno {
    ($e:expr) => {
        if $e {
            return Err(io::Error::last_os_error());
        }
    };
}

//this should be in libc, but isn't
#[derive(Copy, Clone)]
#[repr(C)]
union EpollData {
    ptr: * mut libc::c_void,
    fd: libc::c_int,
    u32: u32,
    u64: u64,
}

impl EpollData {
    const fn new(fd: RawFd) -> Self {
        EpollData{fd}
    }

    //TODO: make const when https://github.com/rust-lang/rust/issues/51909
    // is resolved
    fn get(self) -> RawFd {
        // Safety: this library only reads from and writes to epoll_data.fd
        // the other fields are included only for layout compatibility
        unsafe{self.fd}
    }
}

// Debug like an i32
impl Debug for EpollData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fd = self.get();
        fd.fmt(f)
    }
}

impl PartialEq for EpollData {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl Eq for EpollData {}

impl Hash for EpollData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.get().hash(state);
    }
}

/// An event, such as that a file is available for reading.
/// Transmute compatible with `libc::epoll_event`
#[repr(C, packed)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EpollEvent {
    pub events: Opts,
    fd: EpollData,
}

impl EpollEvent {
    pub const fn zeroed() -> Self {
        Self::new(Opts::empty(), 0)
    }

    const fn new(opts: Opts, fd: RawFd) -> Self {
        EpollEvent{events: opts, fd:EpollData::new(fd)}
    }

    /// The RawFd that this event is associated with.
    ///
    /// For example if a Vec of sockets were added to an epoll instance,
    /// this is the fd of the socket that is ready.
    /// Do not create an [`OwnedRawFd`] (including File) out of this, as closing it
    /// will close the file that was added to Epoll.
    pub fn fd(&self) -> RawFd {
        // Safety: every bit pattern is a valid libc::c_int, so this is always safe
        // Furthermore, this library mantains as an invariant that self.fd.fd refers
        // to an open file
        
        #[cfg(debug_assert)]
        {
            let fd = self.fd.get();
            // valid fds are always non-negative
            debug_assert!( fd >= 0 );
            // Safety: every bit pattern is a valid u64
            let u64 = unsafe{ self.fd.u64 };
            // make sure padding bits are zeroed
            const PADDING_SIZE:usize = size_of::<EpollData>() - size_of::<libc::c_int>();
            debug_assert_eq!(&u64.to_be_bytes()[0..PADDING_SIZE], &[0; PADDING_SIZE]);
        }
        self.fd.get()
    }
}

/// Rust abstraction atop linux's epoll interface.
/// Wrapper type around an epoll file descriptor. Performs proper cleanup on drop.
///
/// ```rust, no_run
/// use std::{time::Duration, fs::File};
/// use std::os::unix::io::AsRawFd;
/// use epoll_rs::{Epoll, Opts, EpollEvent};
///
/// # fn main() -> std::io::Result<()> {
/// let mut epoll = Epoll::new()?;
/// let file = File::open("/")?;
/// let token = epoll.add(file, Opts::IN)?;
/// // add other files...
/// let mut buf = [EpollEvent::zeroed(); 10];
/// let events = epoll.wait_timeout(&mut buf, Duration::from_millis(50))?;
/// for event in events {
///     if token.fd() == event.fd() {
///         println!("File ready for reading");
///     } else {
///         println!("File not ready for reading");
///     }
/// }
/// epoll.remove(token); // this cleanup is performed when epoll goes out of scope
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Epoll {
    epoll_fd: RawFd
}

impl AsRawFd for Epoll {
    fn as_raw_fd(&self) -> unix::io::RawFd {
        self.epoll_fd
    }
}

impl IntoRawFd for Epoll {
    fn into_raw_fd(self) -> RawFd {
        self.epoll_fd
    }
}

impl FromRawFd for Epoll {
    /// Safety: super unsafe. Use only if fd came from this library.
    /// Otherwise, you need to make sure all fds added to this epoll have
    /// `epoll_data` that contains their own fd, that all added fds are open
    /// and that fd is open and refers to an epoll file description.
    unsafe fn from_raw_fd(fd: RawFd) -> Self {
        Epoll{epoll_fd: fd}
    }
}

impl Drop for Epoll {
    fn drop(&mut self) {
        // Safety: this library mantains as an invariant that self.epoll_fd
        // refers to a valid, open file, but libc::close is safe to call on
        // invalid/closed file descriptors too (it returns -1 and sets errno)
        unsafe {libc::close(self.epoll_fd)};
    }
}

/// A trait for all structs that wrap a unix file descriptor.
///
/// This trait is specifically not implemented for RawFd itself, since that
/// would safely allow the use of fds that don't refer to an open file.
/// TODO: replace with `Into<OwnedFd>` when !#[feature(io_safety)] lands
/// <https://github.com/rust-lang/rust/issues/87074>
pub trait OwnedRawFd: AsRawFd + IntoRawFd + FromRawFd {}

impl OwnedRawFd for std::fs::File {}
impl OwnedRawFd for net::TcpListener {}
impl OwnedRawFd for net::TcpStream {}
impl OwnedRawFd for net::UdpSocket {}
impl OwnedRawFd for unix::net::UnixDatagram {}
impl OwnedRawFd for unix::net::UnixListener {}
impl OwnedRawFd for unix::net::UnixStream {}
// impl OwnedRawFd for io::Stderr {}
// impl OwnedRawFd for io::Stdin {}
// impl OwnedRawFd for io::Stdout {}
// impl OwnedRawFd for std::process::ChildStderr {}
// impl OwnedRawFd for std::process::ChildStdin {}
// impl OwnedRawFd for std::process::ChildStdout {}
// impl OwnedRawFd for io::StderrLock<'_> {}
// impl OwnedRawFd for io::StdinLock<'_> {}
// impl OwnedRawFd for io::StdoutLock<'_> {}
impl OwnedRawFd for Epoll {}

impl Epoll {
    pub fn new() -> io::Result<Self> {
        // Safety: Always safe. We're passing flags and getting an fd back
        let fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        then_errno!(fd == -1);
        Ok(Epoll {epoll_fd: fd})
    }

    // panic if token comes from a different epoll instance
    #[cfg(debug_assertions)]
    fn check_token<F: OwnedRawFd>(&self, token: &Token<'_, F>) {
        assert_eq!(self as *const _, token.epoll as *const _);
    }

    /// Add a file-like struct to the epoll instance
    ///
    /// The returned token can be ignored if you don't need to distinguish
    /// which file is ready, but dropping the token closes the added file
    pub fn add<'a, 'b:'a, F: OwnedRawFd>(
        &'b self,
        file: F,
        opts: Opts,
    ) -> io::Result<Token<'a, F>> {
        // Safety: lifetime bounds on function declaration keep this safe
        unsafe { self.add_raw_fd(file.into_raw_fd(), opts) }
    }

    /// Remove a previously added file-like struct from this epoll instance
    ///
    /// No new events will be deilvered referring to this token even if the 
    /// event occurs before removal. Consumes the token, returning the contained file,
    /// since the file it is associated with is no longer in this epoll instance
    pub fn remove<'a, 'b:'a, F: OwnedRawFd>(
        &'b self,
        token: Token<'a, F>,
    ) -> io::Result<F> {
        #[cfg(debug_assertions)]
        self.check_token(&token);
        let empty_event = &mut EpollEvent::zeroed();
        // Safety: empty event (required for early linux kernels) must point to
        // a valid epoll_event struct. This is guaranteed by EpollEvent having
        // the same memory layout as struct epoll_event.
        let res = unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                libc::EPOLL_CTL_DEL,
                token.file.as_raw_fd(),
                empty_event as *mut EpollEvent as *mut _,
            )
        };
        then_errno!(res == -1);
        Ok(token.into_file())
    }

    /// Change the [`Opts`] of a previously added file-like struct
    pub fn modify<'a, 'b: 'a, F: OwnedRawFd>(
        &'b self,
        token: &'a Token<'a, F>,
        opts: Opts,
    ) -> io::Result<()> {
        #[cfg(debug_assertions)]
        self.check_token(token);
        let mut event = EpollEvent::new(opts, token.file.as_raw_fd());
        // Safety: event must point to a valid epoll_event struct
        let res = unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                libc::EPOLL_CTL_MOD,
                token.file.as_raw_fd(),
                &mut event as *mut EpollEvent as *mut _,
            )
        };
        then_errno!(res == -1);
        Ok(())
    }

    // TODO: use uninitalized memory. waiting on stabilization of
    // maybe_uninit_slice, https://github.com/rust-lang/rust/issues/63569
    /// If passed a zero length buf, this function will return Err
    fn wait_maybe_timeout<'a, 'b>(
        &'a self, buf: &'b mut[EpollEvent],
        timeout: Option<Duration>,
    ) -> io::Result<&'b mut[EpollEvent]> {
        let timeout_ms = match timeout {
            Some(t) => t.as_millis().try_into().unwrap_or(i32::MAX),
            None => -1
        };
        let max_events = buf.len().clamp(0, libc::c_int::MAX.try_into().unwrap());
        // Safety: buf_size must be non-zero and <= to the length of self.buf
        // self.buf.as_mut_ptr must point to memory sized and aligned for epoll_events
        let res = unsafe {
            libc::epoll_wait(
                self.epoll_fd,
                buf.as_mut_ptr() as *mut libc::epoll_event,
                max_events as libc::c_int,
                timeout_ms,
            )
        };
        then_errno!(res == -1);
        let len = res.try_into().unwrap();
        Ok(&mut buf[0..len])
    }

    /// Wait indefinetly until at least one event and at most `buf.len()` events occur.
    ///
    /// If passed a zero length buf, this function will return Err
    pub fn wait<'a, 'b>(&'a self, buf: &'b mut[EpollEvent]) -> io::Result<&'b mut [EpollEvent]> {
        self.wait_maybe_timeout(buf, None)
    }

    /// Wait until at least one event and at most `buf.len()` events occur or timeout expires.
    ///
    /// If passed a zero length buf, this function will return Err
    pub fn wait_timeout<'a, 'b>(
        &'a self, 
        buf: &'b mut[EpollEvent],
        timeout: Duration,
    ) -> io::Result<&'b mut [EpollEvent]> {
        self.wait_maybe_timeout(buf, Some(timeout))
    }

    /// Wait indefinetly for one event.
    pub fn wait_one(&self) -> io::Result<EpollEvent> {
        let mut buf = [EpollEvent::zeroed(); 1];
        let res = self.wait(&mut buf);
        res.map(|slice| slice[0])
    }

    /// Wait for one event or until timeout expires.
    ///
    /// Return value of Ok(None) indicates timeout expired
    pub fn wait_one_timeout(&self, timeout: Duration) -> io::Result<Option<EpollEvent>> {
        let mut buf = [EpollEvent::zeroed(); 1];
        let res = self.wait_timeout(&mut buf, timeout);
        res.map(|slice| slice.get(0).copied())
    }

    /// Manually close this epoll instance, handling any potential errors
    ///
    /// Same as drop, only this lets the user deal with errors in closing.
    /// The invariants of this library should mean that close never fails, but
    /// those invariants can be broken with unsafe code.
    pub fn close(&mut self) -> io::Result<()> {
        // Safety: this library mantains as an invariant that self.epoll_fd
        // refers to a valid, open file, but libc::close is safe to call on
        // invalid/closed file descriptors too (it returns -1 and sets errno)
        let res = unsafe {libc::close(self.epoll_fd)};
        then_errno!(res == -1);
        Ok(())
    }

    /// Adds a RawFd to an epoll instance directly
    ///
    /// This is pretty unsafe, prefer [add](Self::add)
    ///
    /// ## Safety
    /// `fd` must refer to a currently open, unowned, file descriptor of type F.
    /// If `fd` refers to a file that is dropped or the fd is closed before this Epoll
    /// instance is, later use of the returned `Token` may modify a different file,
    /// since file descriptors (`RawFd`s) are reused.
    /// The following is notably [io unsound](https://rust-lang.github.io/rfcs/3128-io-safety.html)
    /// ```rust
    /// use epoll_rs::{Epoll, Opts, Token};
    /// use std::{fs::File, io, os::unix::io::{FromRawFd, AsRawFd}};
    /// # fn main() -> io::Result<()> {
    /// let mut epoll = Epoll::new()?;
    /// {
    ///     let stdin = unsafe{File::from_raw_fd(1)};
    ///     let token: Token<File> = unsafe {
    ///         epoll.add_raw_fd(stdin.as_raw_fd(), Opts::IN)?
    ///     };
    /// } // stdin dropped here, fd 1 `libc::close`d, invariants violated
    /// # Ok(())
    /// # }
    /// ```
    /// instead use into_raw_fd to get an unowned RawFd
    /// ```rust
    /// use epoll_rs::{Epoll, Opts, Token};
    /// use std::{fs::File, io, os::unix::io::{FromRawFd, AsRawFd, IntoRawFd}};
    /// # fn main() -> io::Result<()> {
    /// let mut epoll = Epoll::new()?;
    /// {
    ///     let stdin = unsafe{File::from_raw_fd(1)};
    ///     let token: Token<File> = unsafe {
    ///         epoll.add_raw_fd(stdin.into_raw_fd(), Opts::IN)?
    ///     };
    /// } // stdin was consumed by into_raw_fd(), so it's drop code won't be run
    /// # Ok(())
    /// # }
    /// ```
    pub unsafe fn add_raw_fd<'a, 'b:'a, F: OwnedRawFd>(
        &'b self,
        fd: RawFd,
        opts: Opts,
    ) -> io::Result<Token<'a, F>> {
        #[cfg(debug_assertions)]
        let token = Token::new(F::from_raw_fd(fd), self);
        #[cfg(not(debug_assertions))]
        let token = Token::new(F::from_raw_fd(fd));
        let mut event = EpollEvent::new(opts, token.file.as_raw_fd());
        let res = libc::epoll_ctl(
            self.epoll_fd,
            libc::EPOLL_CTL_ADD,
            fd,
            &mut event as *mut _ as *mut libc::epoll_event
        );
        then_errno!(res == -1);

        Ok(token)
    }
}

// Use a doctest because those are allowed to fail at compile time
/// ```compile_fail
/// # use epoll_rs::*;
/// # use std::*;
/// # use time::*;
/// # use fs::*;
/// # fn main() -> std::io::Result<()> {
/// let file = File::open("").unwrap();
/// let token = {
///     let mut epoll = Epoll::new().unwrap();
///     let token = epoll.add(file, Opts::OUT).unwrap();
///     epoll.wait_one_timeout(Duration::from_millis(10)).unwrap();
///     token
/// }; // epoll doesn't live long enough
/// # Ok(())
/// # }
/// ```
//#[cfg(test)]
#[doc(hidden)]
#[allow(unused)] // this test is actually a doctest
fn test_token_lifetime() {}

#[cfg(test)]
mod test {
    use rand::Rng;

    use crate::{Epoll, EpollEvent, Opts, Token};
    use std::collections::HashMap;
    use std::convert::TryInto;
    use std::os::unix::{io::{AsRawFd, FromRawFd}, prelude::RawFd};
    use std::{
        fs::File,
        io::{self, Read, Write},
        thread,
        time::{Duration, Instant},
        mem::{size_of, align_of},
    };

    // Checks that two different types have the same memory representation bit
    // for bit. This is an important guarantee because the linux kernel works
    // with C definitions, and this library reimplements those definitions for
    // convience.
    fn assert_bitwise_eq<T: Sized, U: Sized>(t:T, u:U) {
        assert_eq!(size_of::<T>(), size_of::<U>(), "can't be bitwise equal if different sizes");
        let left_ptr = &t as *const T as *const u8;
        let right_ptr = &u as *const U as *const u8;
        for byte_idx in 0_isize..(size_of::<T>().try_into().unwrap()) {
            // Safety: we know size of T == size of U and we read bytes only within this range
            let left_byte = unsafe { *left_ptr.offset(byte_idx)};
            let right_byte = unsafe { *right_ptr.offset(byte_idx)};
            assert_eq!(left_byte, right_byte, "Byte number {} is different", byte_idx);
        }
    }

    #[test]
    fn test_epoll_event_equivalence() {
        assert_eq!(size_of::<libc::epoll_event>(), size_of::<EpollEvent>());
        assert_eq!(align_of::<libc::epoll_event>(), align_of::<EpollEvent>());

        let libc_event= libc::epoll_event{events: libc::EPOLLOUT as u32, u64: i32::MAX as u64};
        let event = EpollEvent::new(Opts::OUT, i32::MAX);
        assert_bitwise_eq(event, libc_event);
    }

    // Opens a unix pipe and wraps in in Rust `File`s
    //                            read  write
    fn open_pipe() -> io::Result<(File, File)> {
        let (read, write) = {
            let mut pipes = [0 as RawFd; 2];
            // Safety: pipes must be sized and aligned to fit two c_ints/RawFds
            let res = unsafe { libc::pipe2(pipes.as_mut_ptr(), 0) };
            if res != 0 {
                return Err(io::Error::last_os_error());
            }
            (pipes[0], pipes[1])
        };
        // Safety: these files will be the sole owner of the file descriptors
        // since the fds are dropped when this function returns
        let read = unsafe { File::from_raw_fd(read) };
        let write = unsafe { File::from_raw_fd(write) };
        Ok((read, write))
    }

    // Tests that an epoll instance can outlive the token it generates,
    // and that dropping a token nullified pending events, even if the event
    // takes place before the token is dropped
    #[test]
    fn test_epoll_outlives_token() {
        let epoll = Epoll::new().unwrap();
        let (read, mut write) = open_pipe().unwrap();
        write.write(&mut[0]).unwrap();
        {
            // Token immediately discarded
            let _ = epoll.add(read, Opts::IN).unwrap();
        }
        let event = epoll.wait_one_timeout(Duration::from_millis(10));
        assert_eq!(event.unwrap(), None);
    }

    #[test]
    fn test_epoll_wait_read() {
        const MESSAGE: &[u8; 6] = b"abc123";
        fn wait_then_read(file: File) -> Instant {
            let epoll = Epoll::new().unwrap();
            let mut tok = epoll.add(file, Opts::IN).unwrap();
            let event = epoll.wait_one().unwrap();
            assert_eq!(
                event,
                EpollEvent::new(Opts::IN, tok.fd())
            );
            let read_instant = Instant::now();
            let mut buf = [0_u8; 100];
            tok.file_mut().read(&mut buf).unwrap();
            assert_eq!(MESSAGE, &buf[0..MESSAGE.len()]);
            read_instant
        }

        let (read, mut write) = open_pipe().unwrap();
        let th = thread::spawn(move || wait_then_read(read));
        thread::sleep(Duration::from_millis(120));
        write.write(MESSAGE).unwrap();
        let instant = th.join().unwrap();
        let elapsed = instant.elapsed();
        assert!(elapsed < Duration::from_millis(1), "elapsed: {:?}", elapsed);
    }

    #[test]
    fn test_timeout() {
        let (read, _write) = open_pipe().unwrap();
        let epoll = Epoll::new().unwrap();
        epoll.add(read, Opts::IN).unwrap();
        for &wait_ms in [0_u64, 30, 100].iter() {
            let start_wait = Instant::now();
            let event = epoll.wait_one_timeout(Duration::from_millis(wait_ms)).unwrap();
            assert_eq!(event, None);
            let elapsed = start_wait.elapsed();
            assert!(
                elapsed > Duration::from_millis(wait_ms),
                "elapsed: {:?}",
                elapsed
            );
            assert!(
                elapsed < Duration::from_millis(wait_ms + 1),
                "elapsed: {:?}",
                elapsed
            );
        }
    }

    #[test]
    fn test_hup() {
        let (read, write) = open_pipe().unwrap();
        let read_fd = read.as_raw_fd();
        let epoll = Epoll::new().unwrap();
        // no need to epoll.add(Opts::HUP) - it is added by default
        let _token = epoll.add(read, Opts::empty()).unwrap();
        drop(write);
        let event = epoll.wait_one_timeout(Duration::from_millis(10)).unwrap().unwrap();
        assert_eq!(
            event,
            EpollEvent::new(Opts::HUP, read_fd)
        )
    }

    #[test]
    fn test_wait_many() {
        // open a bunch of pipes
        const NUM_PIPES:usize = 20;
        const MESSAGE: &[u8;12] = b"test message";
        let (reads, mut writes):(Vec<File>, Vec<File>) = 
            (0..NUM_PIPES)
            .map(|_| open_pipe().unwrap())
            .unzip();
        let epoll = Epoll::new().unwrap();
        // Add read ends of pipes to an epoll instance
        let mut tokens: HashMap<RawFd, (usize, Token<File>)> = reads
            .into_iter()
            .map(|read| epoll.add(read, Opts::IN).unwrap())
            .enumerate()
            .map(|(idx, tok)| (tok.fd(), (idx,tok)))
            .collect();

        // Write to a random pipe in `writes`
        let secret_rand = {
            let mut rng = rand::thread_rng();
            let rand = rng.gen_range(0..NUM_PIPES);
            eprintln!("Writing to pipe {}", rand);
            assert_eq!(epoll.wait_one_timeout(Duration::from_millis(0)).unwrap(), None);
            writes[rand].write(MESSAGE).unwrap();
            rand
        };

        // Epoll.wait to find out which pipe was written to
        let event = epoll.wait_one_timeout(Duration::from_millis(10)).unwrap().unwrap();
        let (idx, token) = tokens.get_mut(&event.fd()).unwrap();

        let mut buf = [0; MESSAGE.len()];
        token.file_mut().read(&mut buf).unwrap();
        assert_eq!(&buf, MESSAGE);
        assert_eq!(*idx, secret_rand);
    }

    // removing a file nullifies pending events
    #[test]
    fn test_remove_ordering() {
        let (read, mut write) = open_pipe().unwrap();
        let epoll = Epoll::new().unwrap();
        let token = epoll.add(read, Opts::IN).unwrap();
        write.write(b"message in a bottle").unwrap();
        epoll.remove(token).unwrap();
        let event = epoll.wait_one_timeout(Duration::from_millis(10)).unwrap();
        assert_eq!(event, None);
    }

    // test that different types can be added to the same epoll instance
    #[test]
    fn test_add_different_types() {
        let (read, _write) = open_pipe().unwrap();
        let localhost = std::net::Ipv4Addr::new(127, 0, 0, 1);
        let socket = std::net::UdpSocket::bind((localhost, 23456)).unwrap();
        let epoll = Epoll::new().unwrap();
        let _pipe_token = epoll.add(read, Opts::IN).unwrap();
        let _sock_token = epoll.add(socket, Opts::IN).unwrap();
    }
}
