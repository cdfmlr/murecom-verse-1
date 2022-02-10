package ncm

import (
	"encoding/json"
	"io/ioutil"
	"os"
)

var Config struct {
	NcmClient []struct {
		Phone       string
		PasswordMD5 string
		Server      string
	} `json:"client"`
	DB string `json:"db"`
}

// InitConfig read the config file
func InitConfig(file string) {
	f, err := os.Open(file)
	must(err)
	j, err := ioutil.ReadAll(f)
	must(err)
	err = json.Unmarshal(j, &Config)
	must(err)
}

// panic if error
func must(err error) {
	if err != nil {
		panic("failed to load config: " + err.Error())
	}
}
