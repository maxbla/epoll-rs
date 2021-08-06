use epoll_rs::{Epoll, Opts};
use std::{
    io,
    net::{Ipv4Addr, UdpSocket},
};

fn main() -> io::Result<()> {
    let localhost = Ipv4Addr::new(127, 0, 0, 1);
    let socket1 = UdpSocket::bind((localhost, 34567))?;
    let socket2 = UdpSocket::bind((localhost, 34568))?;
    let send_socket = UdpSocket::bind((localhost, 34569))?;

    let mut epoll = Epoll::new()?;
    let epoll = &mut epoll;
    let sock_tok_1 = epoll.add(&socket1, Opts::IN)?;
    let sock_tok_2 = epoll.add(&socket2, Opts::IN)?;

    // write to a random socket, then "forget" which socket was written to
    // simulate an outside network connection
    if rand::random() {
        println!("Writing to socket 1... Shhh don't tell the rest of the program");
        send_socket.connect(socket1.local_addr()?)?;
        send_socket.send(b"Hello, Network!")?;
    } else {
        println!("Writing to socket 2... Shhh don't tell the rest of the program");
        send_socket.connect(socket2.local_addr()?)?;
        send_socket.send(b"Hello, Network!")?;
    }

    let events = epoll.wait()?;
    let first_event = events[0].clone();

    if first_event == sock_tok_1 {
        println!("Wrote to socket 1");
    } else if first_event == sock_tok_2 {
        println!("Wrote to socket 2")
    }
    Ok(())
}
