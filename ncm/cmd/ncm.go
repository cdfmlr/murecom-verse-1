package main

import (
	"flag"
	"fmt"
	"github.com/cdfmlr/murecom-verse-1/ncm"
	"net/http"
	_ "net/http/pprof"
)

var configFile = flag.String("config", "config.json", "/path/to/config/file")

func main() {
	flag.Parse()

	// init
	ncm.InitAll(*configFile)

	// profile
	profile := ncm.Config.Profile
	if profile != "" {
		// http://{ip:port}/debug/pprof/
		go func() {
			fmt.Printf(fmt.Sprintf("Profile: http://%v/debug/pprof", profile))
			panic(http.ListenAndServe(profile, nil))
		}()
	}

	// run
	ncm.Master()
}
