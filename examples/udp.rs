use epoll_rs::{Opts, new_epoll};
use std::{
    fs::File,
    io::{self, Read},
    net::{Ipv4Addr, UdpSocket},
    os::unix::io::{AsRawFd, FromRawFd},
};

fn to_file<F:AsRawFd>(f:&F) -> File {
    unsafe {File::from_raw_fd(f.as_raw_fd())}
}

fn main() -> io::Result<()> {
    let localhost = Ipv4Addr::new(127, 0, 0, 1);
    let socket1 = UdpSocket::bind((localhost, 34567))?;
    let socket2 = UdpSocket::bind((localhost, 34568))?;
    let send_socket = UdpSocket::bind((localhost, 34569))?;

    let socket_1_file = to_file(&socket1);
    let socket_2_file = to_file(&socket2);

    let mut epoll = new_epoll!()?;
    let epoll = &mut epoll;
    let sock_tok_1 = epoll.add_file(&socket_1_file, Opts::IN)?;
    let sock_tok_2 = epoll.add_file(&socket_2_file, Opts::IN)?;

    // write to a random socket, then "forget" which socket was written to
    // simulate an outside network connection
    {
        let rand_bool = {
            let mut buf: [u8; 1] = [0; 1];
            let mut random_source = File::open("/dev/urandom")?;
            random_source.read(&mut buf)?;
            if buf[0] & 1 == 1 {
                true
            } else {
                false
            }
        };
        if rand_bool {
            println!("Writing to socket 1... Shhh don't tell the rest of the program");
            send_socket.connect(socket1.local_addr()?)?;
            send_socket.send(b"Hello, Network!")?;
        } else {
            println!("Writing to socket 2... Shhh don't tell the rest of the program");
            send_socket.connect(socket2.local_addr()?)?;
            send_socket.send(b"Hello, Network!")?;
        }
    }

    let events = epoll.wait()?;
    let first_event = events[0];

    if first_event == sock_tok_1 {
        println!("Wrote to socket 1");
    } else if first_event == sock_tok_2 {
        println!("Wrote to socket 2")
    }
    Ok(())
}