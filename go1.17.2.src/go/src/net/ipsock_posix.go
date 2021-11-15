// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || (js && wasm) || linux || netbsd || openbsd || solaris || windows
// +build aix darwin dragonfly freebsd js,wasm linux netbsd openbsd solaris windows

package net

import (
	"context"
	"internal/poll"
	"runtime"
	"syscall"
)

// probe probes IPv4, IPv6 and IPv4-mapped IPv6 communication
// capabilities which are controlled by the IPV6_V6ONLY socket option
// and kernel configuration.
//
// Should we try to use the IPv4 socket interface if we're only
// dealing with IPv4 sockets? As long as the host system understands
// IPv4-mapped IPv6, it's okay to pass IPv4-mapped IPv6 addresses to
// the IPv6 interface. That simplifies our code and is most
// general. Unfortunately, we need to run on kernels built without
// IPv6 support too. So probe the kernel to figure it out.
func (p *ipStackCapabilities) probe() {
	s, err := sysSocket(syscall.AF_INET, syscall.SOCK_STREAM, syscall.IPPROTO_TCP)
	switch err {
	case syscall.EAFNOSUPPORT, syscall.EPROTONOSUPPORT:
	case nil:
		poll.CloseFunc(s)
		p.ipv4Enabled = true
	}
	var probes = []struct {
		laddr TCPAddr
		value int
	}{
		// IPv6 communication capability
		{laddr: TCPAddr{IP: ParseIP("::1")}, value: 1},
		// IPv4-mapped IPv6 address communication capability
		{laddr: TCPAddr{IP: IPv4(127, 0, 0, 1)}, value: 0},
	}
	switch runtime.GOOS {
	case "dragonfly", "openbsd":
		// The latest DragonFly BSD and OpenBSD kernels don't
		// support IPV6_V6ONLY=0. They always return an error
		// and we don't need to probe the capability.
		probes = probes[:1]
	}
	for i := range probes {
		s, err := sysSocket(syscall.AF_INET6, syscall.SOCK_STREAM, syscall.IPPROTO_TCP)
		if err != nil {
			continue
		}
		defer poll.CloseFunc(s)
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, probes[i].value)
		sa, err := probes[i].laddr.sockaddr(syscall.AF_INET6)
		if err != nil {
			continue
		}
		if err := syscall.Bind(s, sa); err != nil {
			continue
		}
		if i == 0 {
			p.ipv6Enabled = true
		} else {
			p.ipv4MappedIPv6Enabled = true
		}
	}
}

// 看下这个用于判断地址类型的函数
func favoriteAddrFamily(network string, laddr, raddr sockaddr, mode string) (family int, ipv6only bool) {
	//可以看到，如果直接写明"tcp4"或"tcp6"时会直接返回对应的地址类型
	switch network[len(network)-1] {
	case '4':
		return syscall.AF_INET, false
	case '6':
		return syscall.AF_INET6, true
	}
	//下面用于处理network为"tcp的场景"，可以看出直接指定"tcp4"或"tcp6"时可以提高一些执行效率，但可能影响扩展性
	//如果使用监听模式，且本地没有指定监听地址，需要通过对系统本地地址进行测试来判定使用的IP类型
	if mode == "listen" && (laddr == nil || laddr.isWildcard()) {
		//supportsIPv4map函数用于测试系统是否支持ipv4MappedIPv6功能。如果系统支持该功能，或者不支持IPv4，则使用IPv6
		if supportsIPv4map() || !supportsIPv4() {
			return syscall.AF_INET6, false
		}
		//如果没有指定监听地址，同时不支持ipv4MappedIPv6功能，且支持IPv4，则使用IPv4
		if laddr == nil {
			return syscall.AF_INET, false
		}
		//通过判断IP地址(长度和格式)来判断所使用的IP类型。实现函数定义在src/net/tcpsock_posix.go中
		return laddr.family(), false
	}

	//如果未指定本端和远端地址或明确指定了本端和远端需要的地址类型为IPv4，则使用IPv4。可用于处理"listen"和"dial" mode。
	//"listen" mode下已经处理了laddr == nil的情况，如果laddr非nil，调用family()函数判断IP类型即可获得IP类型
	if (laddr == nil || laddr.family() == syscall.AF_INET) &&
		(raddr == nil || raddr.family() == syscall.AF_INET) {
		return syscall.AF_INET, false
	}
	//其他情况使用IPv6，如仅支持IPv6的场景
	return syscall.AF_INET6, false
}

func internetSocket(ctx context.Context, net string, laddr, raddr sockaddr, sotype, proto int, mode string, ctrlFn func(string, string, syscall.RawConn) error) (fd *netFD, err error) {
	//此处判断mode为"dial"的场景。如果"dial" mode下的远端IP为通配符，则将远端IP转换为本地IP(127.0.0.1或::1),即默认连接本地server。
	if (runtime.GOOS == "aix" || runtime.GOOS == "windows" || runtime.GOOS == "openbsd") && mode == "dial" && raddr.isWildcard() {
		raddr = raddr.toLocal(net)
	}
	//favoriteAddrFamily函数用于判断地址类型为IPv4还是IPv6，主要用于net为"tcp"时的场景。
	family, ipv6only := favoriteAddrFamily(net, laddr, raddr, mode)
	return socket(ctx, net, family, sotype, proto, ipv6only, laddr, raddr, ctrlFn)
}

func ipToSockaddr(family int, ip IP, port int, zone string) (syscall.Sockaddr, error) {
	switch family {
	case syscall.AF_INET:
		if len(ip) == 0 {
			ip = IPv4zero
		}
		ip4 := ip.To4()
		if ip4 == nil {
			return nil, &AddrError{Err: "non-IPv4 address", Addr: ip.String()}
		}
		sa := &syscall.SockaddrInet4{Port: port}
		copy(sa.Addr[:], ip4)
		return sa, nil
	case syscall.AF_INET6:
		// In general, an IP wildcard address, which is either
		// "0.0.0.0" or "::", means the entire IP addressing
		// space. For some historical reason, it is used to
		// specify "any available address" on some operations
		// of IP node.
		//
		// When the IP node supports IPv4-mapped IPv6 address,
		// we allow a listener to listen to the wildcard
		// address of both IP addressing spaces by specifying
		// IPv6 wildcard address.
		if len(ip) == 0 || ip.Equal(IPv4zero) {
			ip = IPv6zero
		}
		// We accept any IPv6 address including IPv4-mapped
		// IPv6 address.
		ip6 := ip.To16()
		if ip6 == nil {
			return nil, &AddrError{Err: "non-IPv6 address", Addr: ip.String()}
		}
		sa := &syscall.SockaddrInet6{Port: port, ZoneId: uint32(zoneCache.index(zone))}
		copy(sa.Addr[:], ip6)
		return sa, nil
	}
	return nil, &AddrError{Err: "invalid address family", Addr: ip.String()}
}
