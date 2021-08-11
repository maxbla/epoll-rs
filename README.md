# Epoll-rs

## What is it?

Epoll-rs is a difficult to misuse, high-level binding to Linux's epoll interface. It provides the `Epoll` type, which wraps an epoll file descriptor, just like `std::fs::File` wraps normal file descriptors.

## Why

[epoll](https://github.com/nathansizemore/epoll) is too low level. It is a safe wrapper around epoll, so doesn't require the use of unsafe, but has sharp edges, like needing to use `epoll::close` instead of automatically calling close on `drop` and making the API consumer deal exclusively with `RawFd`s and not `File`s.

[Mio](https://github.com/tokio-rs/mio) is complicated because it aims to support multiple platforms, which epoll-rs doesn't.

## How do I use it?

See the examples directory and top level api documentation