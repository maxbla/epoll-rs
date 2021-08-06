//! A rusty wrapper for the epoll syscall that is difficult to misuse.
//!
//! Create a new epoll instance with the [`new_epoll`] macro. Add any struct
//! that implements the [`OwnedRawFd`] trait with [`Epoll::add`].
//!
//! Files added to the epoll instance must live as long as it
//! ```compile_fail
//! use epoll_rs::{Epoll, Opts};
//! let mut epoll = Epoll::new().unwrap();
//! let epoll = &mut epoll;
//! let token = {
//!     let f = std::fs::File::open("/").unwrap();
//!     epoll.add(&f, Opts::IN).unwrap()
//! }; // Error f dropped while still borrowed
//! ```
//!
//! Tokens returned from one epoll instance cannot be used with another instance.
//! A trick to statically guarantee this is to give your epoll objects different
//! buffer sizes, which actually changes the type.
//! ```compile_fail
//! use epoll_rs::{Epoll, Opts};
//! use std::io;
//! let mut epoll1 = Epoll::<1>::with_capacity().unwrap();
//! let mut epoll1 = &mut epoll1;
//! let mut epoll2 = Epoll::<2>::with_capacity().unwrap();
//! let epoll2 = &mut epoll2;
//! let f = std::fs::File::open("/").unwrap();
//! let token1 = (&mut epoll1).add(&f, Opts::IN).unwrap();
//! let res = epoll2.remove(token1); // <- expected closure, found different closure (not a great error message)
//! ```

use bitflags::bitflags;
use libc::epoll_event;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::os::unix::{self, io::AsRawFd, prelude::RawFd};
use std::{convert::TryInto, io, marker::PhantomData, net, time::Duration, mem::size_of};

/// Opaque type used to refer to single files registered with an epoll instance
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Token<'a, const N: usize> {
    #[cfg(debug_assertions)]
    epoll_fd: RawFd,
    fd: RawFd,
    phantom: PhantomData<&'a Epoll<N>>,
}

impl<const N: usize> Token<'_, N> {
    #[cfg(debug_assertions)]
    fn new(fd: RawFd, epoll_fd: RawFd) -> Self {
        Token {
            epoll_fd,
            fd,
            phantom: PhantomData,
        }
    }
    #[cfg(not(debug_assertions))]
    fn new(fd: RawFd) -> Self {
        Token {
            fd,
            phantom: PhantomData,
        }
    }
}

bitflags! {
    /// Options used in [adding](crate::Epoll::add) a file or
    /// [modifying](crate::Epoll::modify) a previously added file
    ///
    /// Bitwise or (`|`) these together to control multiple options
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
    pub(crate) const fn new(fd: RawFd) -> Self {
        EpollData{fd}
    }

    pub(crate) fn get(self) -> RawFd {
        // Safety: this library only reads from and writes to epoll_data.fd
        // the other fields are only included for layout compatibility
        unsafe{self.fd}
    }
}

// Debug like an i32
impl Debug for EpollData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Safety: this library only reads from and writes to epoll_data.fd
        // the other fields are only included for layout compatibility
        let fd = self.get();
        fd.fmt(f)
    }
}

impl PartialEq for EpollData {
    fn eq(&self, other: &Self) -> bool {
        // Safety: this library only reads from and writes to epoll_data.fd
        // the other fields are only included for layout compatibility
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
/// Transmute compatible with libc::epoll_event
#[repr(C, packed)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EpollEvent {
    pub events: Opts,
    fd: EpollData,
}

impl EpollEvent {
    const fn zeroed() -> Self {
        Self::new(Opts::empty(), 0)
    }

    const fn new(opts: Opts, fd: RawFd) -> Self {
        EpollEvent{events: opts, fd:EpollData::new(fd)}
    }

    fn fd(&self) -> RawFd {
        // Safety: every bit pattern is a valid libc::c_int, so this is always safe
        // Furthermore, this library mantains as an invariant that self.fd.fd refers
        // to an open file
        let fd = self.fd.get();
        #[cfg(debug_assert)]
        {
            // valid fds are always non-negative
            debug_assert!( fd >= 0 );
            // Safety: every bit pattern is a valid u64
            let u64 = unsafe{ self.fd.u64 };
            // make sure padding bits are zeroed
            const PADDING_SIZE:usize = size_of::<EpollData>() - size_of::<libc::c_int>();
            debug_assert_eq!(&u64.to_be_bytes()[0..PADDING_SIZE], &[0; PADDING_SIZE]);
        }
        fd
    }
}

// impl<const N: usize> PartialEq<Token<'_, N>> for EpollEvent {
//     fn eq(&self, other: &Token<N>) -> bool {
//         other.fd == self.fd()
//     }
// }

// impl<const N: usize> PartialEq<EpollEvent> for Token<'_, N> {
//     fn eq(&self, other: &EpollEvent) -> bool {
//         other.fd() == self.fd
//     }
// }

impl<const N: usize> Drop for Epoll<N> {
    fn drop(&mut self) {
        // Safety: this library mantains as an invariant that self.epoll_fd
        // refers to a valid, open file, but libc::close is safe to call on
        // invalid/closed file descriptors too (it returns -1 and sets errno)
        unsafe { libc::close(self.epoll_fd) };
    }
}

/// Rust abstraction atop linux's epoll interface.
/// Wrapper type around an epoll file descriptor. Performs proper cleanup on drop.
///
/// ```rust, no_run
/// let mut epoll = epoll_rs::Epoll::new().unwrap();
/// let epoll = &mut epoll;
/// let input = std::io::stdin();
/// let token = epoll.add(&input, epoll_rs::Opts::IN).unwrap();
/// // add other files...
/// let events = epoll.wait_timeout(std::time::Duration::from_millis(50)).unwrap();
/// for event in events {
///     if &token == event {
///         println!("Got input from stdin");
///     } else {
///         println!("Got input from elsewhere");
///     }
/// }
/// epoll.remove(token); // this cleanup is also performed when epoll goes out of scope
/// ```
/// ## Why do I need to take a reference to the epoll struct to use it?
/// So that this library can prove that it outlives files added to it
#[derive(Debug)]
pub struct Epoll<const N: usize> { //TODO: change N to NonZeroUsize
    epoll_fd: RawFd,
    buf: [EpollEvent; N],
}

impl<const N: usize> AsRawFd for Epoll<N> {
    fn as_raw_fd(&self) -> unix::io::RawFd {
        self.epoll_fd
    }
}

/// A trait for all structs that wrap a unix file descriptor.
///
/// This trait is specifically not implemented for RawFd itself, since that
/// would allow the use of fds that don't refer to an open file.
pub trait OwnedRawFd: AsRawFd {}

impl OwnedRawFd for std::fs::File {}
impl OwnedRawFd for io::Stderr {}
impl OwnedRawFd for io::Stdin {}
impl OwnedRawFd for io::Stdout {}
impl OwnedRawFd for net::TcpListener {}
impl OwnedRawFd for net::TcpStream {}
impl OwnedRawFd for net::UdpSocket {}
impl OwnedRawFd for unix::net::UnixDatagram {}
impl OwnedRawFd for unix::net::UnixListener {}
impl OwnedRawFd for unix::net::UnixStream {}
impl OwnedRawFd for std::process::ChildStderr {}
impl OwnedRawFd for std::process::ChildStdin {}
impl OwnedRawFd for std::process::ChildStdout {}
impl OwnedRawFd for io::StderrLock<'_> {}
impl OwnedRawFd for io::StdinLock<'_> {}
impl OwnedRawFd for io::StdoutLock<'_> {}
impl<const N: usize> OwnedRawFd for Epoll<N> {}

impl Epoll<50> {
    pub fn new() -> io::Result<Self> {
        Epoll::with_capacity()
    }
}

impl<const N: usize> Epoll<N> {
    /// Returns the number of epoll events that can be stored by this Epoll instance
    pub const fn capacity() -> usize {
        N
    }

    pub fn token_from_event(&self, event: EpollEvent) -> Token<'static, N> {
        Token::new(event.fd(), self.epoll_fd)
    }

    pub fn with_capacity() -> io::Result<Self> {
        // Safety: Always safe. We're passing flags and getting an fd back
        let fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        then_errno!(fd == -1);
        const ZEROED_EVENT: EpollEvent = EpollEvent::zeroed();
        Ok(Epoll {
            epoll_fd: fd,
            buf: [ZEROED_EVENT; N],
        })
    }

    /// Adds a RawFd to an epoll instance directly
    ///
    /// This is pretty unsafe, prefer [add](Self::add)
    ///
    /// ## Safety
    /// `fd` must refer to a currently open file that does not outlive the
    /// Epoll instance. If `fd` refers to a file that is dropped before this Epoll
    /// instance, later use of the returned `Token` may modify a different file,
    /// since file descriptors (`RawFd`s) are reused.
    pub unsafe fn add_raw_fd<'a, 'c>(self: &'c&'a mut Self, fd: RawFd, opts: Opts) -> io::Result<Token<'a, N>> {
        #[cfg(debug_assertions)]
        let token = Token::new(fd, self.epoll_fd);
        #[cfg(not(debug_assertions))]
        let token = Token::new(fd);
        let mut event = epoll_event {
            events: opts.bits(),
            u64: token.fd.try_into().unwrap(),
        };
        let res = libc::epoll_ctl(self.epoll_fd, libc::EPOLL_CTL_ADD, fd, &mut event as *mut _);
        then_errno!(res == -1);

        Ok(token)
    }

    /// Add a file-like struct to the epoll instance
    ///
    /// The returned token can be ignored if you don't need to distinguish
    /// which file is ready.
    pub fn add<'a, 'b: 'a, 'c, F: OwnedRawFd>(
        self: &'c &'a mut Self,
        file: &'b F,
        opts: Opts,
    ) -> io::Result<Token<'a, N>> {
        // Safety: lifetime bounds on function declaration keep this safe
        unsafe { self.add_raw_fd(file.as_raw_fd(), opts) }
    }

    /// Remove a previously added file-like struct from this epoll instance
    ///
    /// No new events will be deilvered referring to this token. Consumes the
    /// token, since the file it is associated with is no longer in this epoll
    /// instance
    pub fn remove<'a>(&'a mut self, token: Token<'a, N>) -> io::Result<()> {
        debug_assert_eq!(self.epoll_fd, token.epoll_fd);
        let empty_event = &mut EpollEvent::zeroed();
        // Safety: empty event (required for early linux kernels) must point to
        // a valid epoll_event struct. This is guaranteed by EpollEvent having
        // the same memory layout as struct epoll_event.
        let res = unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                libc::EPOLL_CTL_DEL,
                token.fd,
                empty_event as *mut EpollEvent as *mut _,
            )
        };
        then_errno!(res == -1);
        Ok(())
    }

    /// Change the [`Opts`] of a previously added file-like struct
    pub fn modify(&self, token: &Token<N>, opts: Opts) -> io::Result<()> {
        debug_assert_eq!(self.epoll_fd, token.epoll_fd);
        let mut event = EpollEvent::new(opts, token.fd);
        // Safety: event must point to a valid epoll_event struct
        let res = unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                libc::EPOLL_CTL_MOD,
                token.fd,
                &mut event as *mut EpollEvent as *mut _,
            )
        };
        then_errno!(res == -1);
        Ok(())
    }

    /// Wait indefinetly for at least one event to occur
    pub fn wait(&mut self) -> io::Result<&[EpollEvent]> {
        self.wait_maybe_timeout(None, None)
    }

    /// Wait until at least one event occurs or timeout expires
    pub fn wait_timeout(&mut self, timeout: Duration) -> io::Result<&[EpollEvent]> {
        self.wait_maybe_timeout(Some(timeout), None)
    }

    /// Wait indefinetly, or until at least one event occurs, with a maximum of `lim` events
    ///
    /// If lim < [capacity](crate::Epoll::capacity), at most capacity events
    /// are returned
    pub fn wait_limit(&mut self, event_limit: usize) -> io::Result<&[EpollEvent]> {
        self.wait_maybe_timeout(None, Some(event_limit))
    }

    /// Wait until at least one event occurs or timeout expires, with a miximum of `lim` events
    ///
    /// If lim > [capacity](crate::Epoll::capacity), at most capacity events
    /// are returned
    pub fn wait_timeout_limit(
        &mut self,
        timeout: Duration,
        event_limit: usize,
    ) -> io::Result<&[EpollEvent]> {
        self.wait_maybe_timeout(Some(timeout), Some(event_limit))
    }

    fn wait_maybe_timeout(
        &mut self,
        timeout: Option<Duration>,
        event_limit: Option<usize>,
    ) -> io::Result<&[EpollEvent]> {
        let timeout_ms = match timeout {
            Some(t) => t.as_millis().try_into().unwrap_or(i32::MAX),
            None => -1
        };
        let max_events = match event_limit {
            Some(lim) if lim <= self.buf.len() => lim,
            _ => self.buf.len()
        };
        // Safety: buf_size must be non-zero and <= to the length of self.buf
        // self.buf.as_mut_ptr must point to memory sized and aligned for epoll_events
        let res = unsafe {
            libc::epoll_wait(
                self.epoll_fd,
                std::mem::transmute(self.buf.as_mut_ptr()),
                max_events as libc::c_int,
                timeout_ms,
            )
        };
        then_errno!(res == -1);
        let res = res as usize;
        Ok(&self.buf[0..res])
    }
}

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

        //let libc_data = epoll_data{u64: u64::MAX};
        let libc_event= libc::epoll_event{events: libc::EPOLLOUT as u32, u64: i32::MAX as u64};

        let event = EpollEvent::new(Opts::OUT, i32::MAX);
        assert_bitwise_eq(event, libc_event);
    }

    // Use a doctest because those are allowed to fail at compile time
    /// ```compile_fail
    /// # use epoll_rs::*;
    /// # use std::*;
    /// # use time::*;
    /// # use fs::*;
    /// let file = File::open("/").unwrap();
    /// let token = {
    ///     let mut epoll = new_epoll!().unwrap();
    ///     let mut epoll = &mut epoll;
    ///     let token = epoll.add_file(&file, Opts::OUT).unwrap();
    ///     epoll.wait_timeout(Duration::from_millis(10)).unwrap();
    ///     token
    /// };
    /// ```
    #[doc(hidden)]
    #[allow(unused)] // this test is actually a doctest
    fn test_token_lifetime() {}

    // Opens a unix pipe and wraps in in Rust `File`s
    //                            read  write
    fn open_pipe() -> io::Result<(File, File)> {
        let (read, write) = {
            let mut pipes = [0 as RawFd; 2];
            // Safety: pipes must be sized and aligned to fit two c_ints/RawFds
            let res = unsafe { libc::pipe2(pipes.as_mut_ptr(), 0) };
            if res == -1 {
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

    #[test]
    fn test_epoll_wait_read() {
        const MESSAGE: &[u8; 6] = b"abc123";
        fn wait_then_read(mut file: File) -> Instant {
            let mut epoll = Epoll::new().unwrap();
            let _tok = (&mut epoll).add(&file, Opts::IN).unwrap();
            let events = epoll.wait().unwrap();
            assert_eq!(
                events[0],
                EpollEvent::new(Opts::IN, file.as_raw_fd())
            );
            let read_instant = Instant::now();
            let mut buf = [0_u8; 100];
            file.read(&mut buf).unwrap();
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
    fn test_epoll_wait_multiple_events() {
        const MESSAGE_1: &[u8; 6] = b"abc123";
        const MESSAGE_2: &[u8; 6] = b"def456";
        fn wait_then_read(mut file: File) -> Instant {
            let mut epoll = Epoll::new().unwrap();
            let _tok = (&mut epoll).add(&file, Opts::IN).unwrap();
            let event1 = {
                let events = epoll.wait_limit(1).unwrap();
                events[0].clone()
            };
            let events2 = epoll.wait_limit(1).unwrap();
            assert_eq!(
                event1,
                EpollEvent::new(Opts::IN, file.as_raw_fd()),
            );
            assert_eq!(
                events2[0],
                EpollEvent::new(Opts::IN, file.as_raw_fd())
            );
            let read_instant = Instant::now();
            let mut buf = [0_u8; MESSAGE_1.len() + MESSAGE_2.len()];
            file.read(&mut buf).unwrap();
            assert_eq!(MESSAGE_1, &buf[0..MESSAGE_1.len()]);
            read_instant
        }

        let (read, mut write) = open_pipe().unwrap();
        let th = thread::spawn(move || wait_then_read(read));
        thread::sleep(Duration::from_millis(120));
        write.write(MESSAGE_1).unwrap();
        write.write(MESSAGE_2).unwrap();
        let instant = th.join().unwrap();
        let elapsed = instant.elapsed();
        assert!(elapsed < Duration::from_millis(1), "elapsed: {:?}", elapsed);
    }

    #[test]
    fn test_timeout() {
        let (read, write) = open_pipe().unwrap();
        let mut epoll = Epoll::new().unwrap();
        let epoll = &mut epoll;
        epoll.add(&read, Opts::IN).unwrap();
        for &wait_ms in [0_u64, 30, 100].iter() {
            let start_wait = Instant::now();
            let _events = epoll.wait_timeout(Duration::from_millis(wait_ms)).unwrap();
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
        // don't drop write until after tests
        drop(write);
    }

    #[test]
    fn test_hup() {
        let (read, write) = open_pipe().unwrap();
        let mut epoll = Epoll::new().unwrap();
        // no need to epoll.add(Opts::HUP) - it is added by default
        let _token = (&mut epoll).add(&read, Opts::RDHUP).unwrap();
        drop(write);
        let buf = epoll.wait_timeout(Duration::from_millis(10)).unwrap();
        assert_eq!(buf.len(), 1);
        assert_eq!(
            buf[0],
            EpollEvent::new(Opts::HUP, read.as_raw_fd())
        )
    }

    #[test]
    fn test_wait_many() {
        // open a bunch of pipes
        const NUM_PIPES:usize = 20;
        const MESSAGE: &[u8;12] = b"test message";
        let (mut reads, mut writes):(Vec<File>, Vec<File>) = (0..NUM_PIPES).map(|_| open_pipe().unwrap()).unzip();
        let mut epoll = Epoll::new().unwrap();
        // Add read ends of pipes to an epoll instance
        let epoll = &mut epoll;
        let tokens: Vec<Token<50>> = reads.iter().map(|read| epoll.add(read, Opts::IN).unwrap()).collect();

        {
            // write to a random pipe
            let mut rng = rand::thread_rng();
            let rand = rng.gen_range(0..NUM_PIPES);
            eprintln!("Writing to pipe {}", rand);
            assert_eq!(epoll.wait_timeout(Duration::from_millis(0)).unwrap().len(), 0);
            writes[rand].write(MESSAGE).unwrap();
        }

        // epoll wait to find out which pipe was written to
        let event = epoll.wait().unwrap().into_iter().next().unwrap();
        let new_event = event.clone();
        let tok = epoll.token_from_event(new_event);
        dbg!(&tok);
        //let (num, _read) = tokens.iter().inspect(|read_tok| {dbg!(read_tok);}).enumerate().find(|&(_, read_tok)| read_tok == &tok).unwrap();
        let mut file = std::mem::ManuallyDrop::new(unsafe{File::from_raw_fd(tok.fd)});
        let mut buf = [0; MESSAGE.len()];
        file.read(&mut buf);
        assert_eq!(&buf, MESSAGE);
        // eprintln!("Saw write from pipe {}", num);
        // assert_eq!(num, rand);

    }
}
