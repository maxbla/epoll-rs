use epoll_rs::{Epoll, Opts};
use std::{io, net::{Ipv4Addr, UdpSocket}};

fn main() -> io::Result<()> {
    let localhost = Ipv4Addr::new(127, 0, 0, 1);
    let socket1 = UdpSocket::bind((localhost, 34567))?;
    let socket2 = UdpSocket::bind((localhost, 34568))?;
    let send_socket = UdpSocket::bind((localhost, 34569))?;

    let mut epoll = Epoll::new()?;
    let epoll = &mut epoll;
    let mut sock_tok_1 = epoll.add(socket1, Opts::IN)?;
    let mut sock_tok_2 = epoll.add(socket2, Opts::IN)?;

    // write to a random socket, then "forget" which socket was written to
    // simulate an external network connection
    if rand::random() {
        println!("Writing to socket 1... Shhh don't tell the rest of the program");
        send_socket.connect(sock_tok_1.file().local_addr()?)?;
        send_socket.send(b"Hello, Network!")?;
    } else {
        println!("Writing to socket 2... Shhh don't tell the rest of the program");
        send_socket.connect(sock_tok_2.file().local_addr()?)?;
        send_socket.send(b"Hello, Network!")?;
    }

    let event = epoll.wait_one()?;

    let mut buf = vec![0;20];
    if event.fd() == sock_tok_1.fd() {
        println!("Observed write on socket 1");
        sock_tok_1.file_mut().recv(&mut buf)?;
    } else if event.fd() == sock_tok_2.fd() {
        println!("Observed write on socket 2");
        sock_tok_2.file_mut().recv(&mut buf)?;
    } else {
        panic!("Neither socket was written to!");
    }
    let recvd_message = String::from_utf8(buf).unwrap();
    println!("{}", recvd_message);

    Ok(())
}
