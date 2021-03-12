use bitflags::bitflags;
use libc::epoll_event;
use std::{cell::UnsafeCell, io, time::Duration};
use std::{convert::TryInto, os::unix::io::AsRawFd};

// TODO: make this crate no_std and use a pure rust syscall implementation

// Like UnsafeCell, only implements Sync
// This would allow concurrent modification by multiple threads, but that causes UB
// This is only used to work around a bug on old versions of Linux
struct SyncUnsafeCell<T> {
    inner: UnsafeCell<T>,
}

unsafe impl<T> Sync for SyncUnsafeCell<T> {}

impl<T> SyncUnsafeCell<T> {
    fn get(&self) -> *mut T {
        self.inner.get()
    }
}

static EMPTY_EVENT: SyncUnsafeCell<epoll_event> = SyncUnsafeCell {
    inner: UnsafeCell::new(epoll_event {
        events: Opts::empty().bits(),
        u64: 0,
    }),
};

/// Opaque type used to refer to single files registered with an epoll instance
///
/// Using a Token with an `Epoll` instance that it did not come from
/// (i.e. with epoll.add_file) is a bug in your program and may result in
/// `io::Errors`, but may also affect seemingly random files
#[derive(Debug, PartialEq, Hash)]
pub struct Token {
    fd: libc::c_int,
}

bitflags! {
    /// Options used in [adding](crate::Epoll::add) a file or
    /// [modifying](crate::Epoll::modify) a previously added file
    ///
    /// Bitwise or (`|`) these together to control multiple options
    pub struct Opts:u32 {
        /// Available for reads
        const IN = libc::EPOLLIN as u32;
        /// Available for writes
        const OUT = libc::EPOLLOUT as u32;
        /// Socket connection closed
        const RDHUP = libc::EPOLLRDHUP as u32;
        /// Exceptional condition (see man 3 poll)
        const PRI = libc::EPOLLPRI as u32;
        const ERR = libc::EPOLLERR as u32;
        /// Hangup. If you register for another event type, this is automatically enabled
        const HUP = libc::EPOLLHUP as u32;
        /// Use edge-triggered notifications
        const ET = libc::EPOLLET as u32;
        const ONESHOT = libc::EPOLLONESHOT as u32;
        const WAKEUP = libc::EPOLLWAKEUP as u32;
        /// Deliver on only one epoll fd (see man epoll)
        const EXCLUSIVE = libc::EPOLLEXCLUSIVE as u32;
    }
}

macro_rules! then_errno {
    ($e:expr) => {
        if $e {
            return Err(io::Error::last_os_error());
        }
    };
}

//Equivalent definition to libc::epoll_event
/// An event, such as that a file is available for reading
#[repr(packed)] // For some reason, this needs to be packed
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EpollEvent {
    pub events: Opts,
    fd: u64,
}

impl From<EpollEvent> for Token {
    fn from(event: EpollEvent) -> Self {
        Token {
            fd: event.fd as i32,
        }
    }
}

impl Drop for Epoll {
    fn drop(&mut self) {
        unsafe { libc::close(self.epoll_fd) };
    }
}

/// Wrapper type around an epoll fd, like the standard library's `File` is a
/// wrapper around normal fds.
#[derive(Debug, PartialEq, Hash)]
pub struct Epoll {
    epoll_fd: i32,
    //TODO: replace with array when const generics are stabilized
    buf: Vec<libc::epoll_event>,
}

impl Epoll {
    /// Like new, but allocates a buffer of size `capacity`
    /// for storing epoll events in Epoll::wait()
    fn with_capacity(capacity: usize) -> io::Result<Self> {
        //TODO: make thie function pub (gated on min-const-generics)
        let fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        then_errno!(fd == -1);
        Ok(Epoll {
            epoll_fd: fd,
            buf: vec![libc::epoll_event { events: 0, u64: 0 }; capacity],
        })
    }

    /// Returns the capacity of the internal buffer storing epoll events
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Open a new epoll fd, but don't add any files to it
    pub fn new() -> io::Result<Self> {
        Self::with_capacity(20)
    }

    fn add_fd(&mut self, fd: i32, opts: Opts) -> io::Result<Token> {
        let token = Token { fd };
        let mut event = epoll_event {
            events: opts.bits(),
            u64: token.fd.try_into().unwrap(),
        };
        let res = unsafe {
            libc::epoll_ctl(self.epoll_fd, libc::EPOLL_CTL_ADD, fd, &mut event as *mut _)
        };
        then_errno!(res == -1);

        Ok(token)
    }

    /// Add a file-like struct to the epoll instance
    ///
    /// The token return type can be ignored if you don't need to distinguish
    /// between files (e.g. you only have one file added)
    pub fn add<F>(&mut self, file: &F, opts: Opts) -> io::Result<Token>
    where
        F: AsRawFd,
    {
        self.add_fd(file.as_raw_fd(), opts)
    }

    /// Remove a previously added file-like struct from this epoll instance
    ///
    /// Consumes the token, since the file it is associated with is no longer
    /// in this epoll instance
    pub fn remove(&mut self, token: Token) -> io::Result<()> {
        let res = unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                libc::EPOLL_CTL_DEL,
                token.fd,
                EMPTY_EVENT.get(),
            )
        };
        then_errno!(res == -1);
        Ok(())
    }

    /// Change the opts of a previously added file-like struct
    pub fn modify(&mut self, token: &Token, opts: Opts) -> io::Result<()> {
        let mut event = epoll_event {
            events: opts.bits(),
            u64: token.fd.try_into().unwrap(),
        };
        let res = unsafe {
            libc::epoll_ctl(
                self.epoll_fd,
                libc::EPOLL_CTL_MOD,
                token.fd,
                &mut event as *mut _,
            )
        };
        then_errno!(res == -1);
        Ok(())
    }

    /// Wait indefinetly, or until at least one event occurs
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
    pub fn wait_limit(&mut self, lim: usize) -> io::Result<&[EpollEvent]> {
        self.wait_maybe_timeout(None, Some(lim))
    }

    /// Wait until at least one event occurs or timeout expires, with a miximum of `lim` events
    ///
    /// If lim < [capacity](crate::Epoll::capacity), at most capacity events
    /// are returned
    pub fn wait_timeout_limit(
        &mut self,
        timeout: Duration,
        lim: usize,
    ) -> io::Result<&[EpollEvent]> {
        self.wait_maybe_timeout(Some(timeout), Some(lim))
    }

    fn wait_maybe_timeout(
        &mut self,
        timeout: Option<Duration>,
        lim: Option<usize>,
    ) -> io::Result<&[EpollEvent]> {
        let timeout_ms = timeout.map(|t| t.as_millis() as i32).unwrap_or(-1);
        let buf_size = lim
            .map(|lim| self.buf.len().min(lim))
            .unwrap_or(self.buf.len());
        let res = unsafe {
            libc::epoll_wait(
                self.epoll_fd,
                self.buf.as_mut_ptr(),
                buf_size as i32,
                timeout_ms,
            )
        };
        then_errno!(res == -1);
        let res = res as usize;
        Ok(unsafe { &*(&self.buf[0..res] as *const [libc::epoll_event] as *const [EpollEvent]) })
    }
}

#[cfg(test)]
mod test {
    use crate::{Epoll, EpollEvent, Opts};
    use std::os::unix::io::{AsRawFd, FromRawFd};
    use std::{
        fs::File,
        io::{self, Read, Write},
        thread,
        time::{Duration, Instant},
    };

    // Opens a unix pipe and wraps in in Rust `File`s
    //                            read  write
    fn open_pipe() -> io::Result<(File, File)> {
        let (read, write) = {
            let mut pipes = [0 as libc::c_int; 2];
            let res = unsafe { libc::pipe2(pipes.as_mut_ptr(), 0) };
            if res == -1 {
                return Err(io::Error::last_os_error());
            }
            (pipes[0], pipes[1])
        };
        let read = unsafe { File::from_raw_fd(read) };
        let write = unsafe { File::from_raw_fd(write) };
        Ok((read, write))
    }

    #[test]
    fn test_epoll_wait_read() {
        const MESSAGE: &[u8; 6] = b"abc123";
        fn wait_then_read(mut file: File) -> Instant {
            let mut epoll = Epoll::new().unwrap();
            let _tok = epoll.add(&file, Opts::IN).unwrap();
            let events = epoll.wait().unwrap();
            assert_eq!(
                events[0],
                EpollEvent {
                    events: Opts::IN,
                    fd: file.as_raw_fd() as u64
                }
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
            let _tok = epoll.add(&file, Opts::IN).unwrap();
            let event1 = {
                let events = epoll.wait_limit(1).unwrap();
                events[0]
            };
            let events2 = epoll.wait_limit(1).unwrap();
            assert_eq!(
                event1,
                EpollEvent {
                    events: Opts::IN,
                    fd: file.as_raw_fd() as u64
                }
            );
            assert_eq!(
                events2[0],
                EpollEvent {
                    events: Opts::IN,
                    fd: file.as_raw_fd() as u64
                }
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
        let (mut epoll, read) = {
            let (read, write) = open_pipe().unwrap();
            let mut epoll = Epoll::new().unwrap();
            // no need to epoll.add(Opts::HUP) - it is added by default
            epoll.add(&read, Opts::RDHUP).unwrap();
            drop(write);
            (epoll, read)
        };
        let buf = epoll.wait_timeout(Duration::from_millis(10)).unwrap();
        assert_eq!(buf.len(), 1);
        assert_eq!(
            buf[0],
            EpollEvent {
                events: Opts::HUP,
                fd: read.as_raw_fd() as u64
            }
        )
    }
}
