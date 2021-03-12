use epoll_rs::{Epoll, Opts, Token};
use std::{
    fs::File,
    io::{self, Read},
    net::{Ipv4Addr, UdpSocket},
};

fn main() -> io::Result<()> {
    let localhost = Ipv4Addr::new(127, 0, 0, 1);
    let socket1 = UdpSocket::bind((localhost, 34567))?;
    let socket2 = UdpSocket::bind((localhost, 34568))?;
    let send_socket = UdpSocket::bind((localhost, 34569))?;

    let mut epoll: Epoll = Epoll::new()?;
    let sock_tok_1 = epoll.add(&socket1, Opts::IN)?;
    let sock_tok_2 = epoll.add(&socket2, Opts::IN)?;

    // write to a random socket, then drop the random number
    // simulates an outside network connection, in which you don't know what
    // socket will be written to
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
            println!("Writing to socket 1. Shhh don't tell the rest of the program");
            send_socket.connect(socket1.local_addr()?)?;
            send_socket.send(b"Hello, Network!")?;
        } else {
            println!("Writing to socket 2. Shhh don't tell the rest of the program");
            send_socket.connect(socket2.local_addr()?)?;
            send_socket.send(b"Hello, Network!")?;
        }
    }

    let events = epoll.wait()?;
    let token: Token = events[0].into();

    if token == sock_tok_1 {
        println!("Wrote to socket 1");
    } else if token == sock_tok_2 {
        println!("Wrote to socket 2")
    }
    Ok(())
}
