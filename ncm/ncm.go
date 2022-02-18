package main

import (
	"fmt"
	"ncm/ncmapi"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path"
	"runtime/pprof"
	"strings"
	"time"
)

const Profile = false
const Debug = false

func main() {
	st := strings.ReplaceAll(time.Now().Local().Format(time.Kitchen), ":", "-")

	if Profile {
		f, _ := os.Create(path.Join(Config.Profile, fmt.Sprintf("%v.prof", st)))
		_ = pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()

		// http://localhost:6060/debug/pprof/
		go func() {
			panic(http.ListenAndServe(":6060", nil))
		}()
	}

	// init
	InitConfig("config.json")
	// ncmapi Init
	err := ncmapi.Init(NcmapiConfigs(), ncmapiLogger{logger})
	if err != nil {
		logger.Fatal("ncmapi Init failed:", err)
	}
	InitDB()
	InitFetcher()

	// run
	Master()

	if Profile {
		fm, _ := os.Create(path.Join(Config.Profile, fmt.Sprintf("%v.mprof", st)))
		_ = pprof.WriteHeapProfile(fm)
		_ = fm.Close()
	}
}
