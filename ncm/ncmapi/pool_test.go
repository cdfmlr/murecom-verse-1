package ncmapi

import (
	"fmt"
	"testing"
)

var testClientConfig ClientConfig

func init() {
	testClientConfig = ClientConfig{
		Phone:       testClient.Phone,
		PasswordMD5: testClient.PasswordMD5,
		Server:      testClient.Server,
	}
}

func TestInit(t *testing.T) {
	type args struct {
		clients      []ClientConfig
		customLogger Logger
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{"no-clients", args{clients: nil, customLogger: nil}, true},
		{"1-client", args{clients: []ClientConfig{testClientConfig}, customLogger: nil}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Init(tt.args.clients, tt.args.customLogger)
			if (err != nil) != tt.wantErr {
				t.Errorf("❌ Init() err=%v, however wantErr=%v", err, tt.wantErr)
			}
			t.Logf("✅ Init() err=%v", err)
		})
	}
}

func TestGetClientDone(t *testing.T) {
	err := Init([]ClientConfig{testClientConfig}, nil)
	if err != nil {
		t.Error("❌", err)
	}

	t.Run("FirstGetClientAndDone", func(t *testing.T) {
		cli := GetClient()
		tpr, err := cli.TopPlaylists(1, 0)
		if err != nil {
			t.Error("❌", err)
		}

		var tpns []string
		for _, p := range tpr.Playlists {
			tpns = append(tpns, p.Name)
		}
		t.Logf("✅ get TopPlaylists from cli successfully: %#v\n", tpns)

		Done(cli)
	})

	t.Run("SecondGetClientNoDone", func(t *testing.T) {
		cli := GetClient()
		tpr, err := cli.TopPlaylists(2, 0)
		if err != nil {
			t.Error("❌", err)
		}

		var tpns []string
		for _, p := range tpr.Playlists {
			tpns = append(tpns, p.Name)
		}
		t.Logf("✅ get TopPlaylists from cli successfully: %#v\n", tpns)
	})

	t.Run("GetClient-New1", func(t *testing.T) {
		cli := GetClient()
		tpr, err := cli.TopPlaylists(3, 0)
		if err != nil {
			t.Error("❌", err)
		}

		var tpns []string
		for _, p := range tpr.Playlists {
			tpns = append(tpns, p.Name)
		}
		t.Logf("✅ get TopPlaylists from cli successfully: %#v\n", tpns)
	})
}

func TestBadClientConfig(t *testing.T) {
	badPhoneConfig := ClientConfig{
		Phone:       "18888888899",
		PasswordMD5: testClientConfig.PasswordMD5,
		Server:      testClientConfig.Server,
	}
	badServerConfig := ClientConfig{
		Phone:       testClientConfig.Phone,
		PasswordMD5: testClientConfig.PasswordMD5,
		Server:      "http://localhost:9999/",
	}
	type args struct {
		clients      []ClientConfig
		customLogger Logger
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{"good-config", args{clients: []ClientConfig{testClientConfig}, customLogger: nil}, false},
		{"bad-config-account", args{clients: []ClientConfig{badPhoneConfig}}, true},
		{"bad-config-server", args{clients: []ClientConfig{badServerConfig}}, true},
		{"good+bads", args{
			clients: []ClientConfig{
				testClientConfig, badPhoneConfig, badServerConfig},
			customLogger: nil}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// init
			err := Init(tt.args.clients, tt.args.customLogger)
			if err != nil {
				t.Errorf("❌ Init() err=%v", err)
			}
			debug("init")

			// get client, do api request
			cli := GetClient()
			debug("GetClient")
			if (cli == nil) != tt.wantErr {
				t.Fatalf("❌ GetClient()=nil, however wantErr=%v", tt.wantErr)
			}
			t.Logf("✅ GetClient()=%v", cli)
			if cli == nil {
				return
			}

			tpr, err := cli.TopPlaylists(1, 0)
			Done(cli)

			if (err != nil) != tt.wantErr {
				t.Fatalf("❌ Use Client err=%v, however wantErr=%v", err, tt.wantErr)
			}
			t.Logf("✅ Use Client err=%v", err)

			if err == nil {
				var tpns []string
				for _, p := range tpr.Playlists {
					tpns = append(tpns, p.Name)
				}
				t.Logf("✅ get TopPlaylists from cli successfully: %#v\n", tpns)
			} else {
				t.Logf("✅ Use Client API result : %#v\n", tpr)
			}
		})
	}
}

func TestNextClientConfig(t *testing.T) {
	var configs []ClientConfig

	for i := 1; i < 5; i++ {
		configs = append(configs, ClientConfig{
			Phone:       fmt.Sprintf("%v%v%v", i, i, i),
			PasswordMD5: "",
			Server:      "",
		})
		t.Run(fmt.Sprintf("configs-%v", i), func(t *testing.T) {
			t.Log("configs:", len(configs))
			err := Init(configs, nil)
			if err != nil {
				t.Errorf("❌ Init() err=%v", err)
			}

			for j := 0; j < 10; j++ {
				c := <-nextClientConfig
				t.Log(j, c.Phone)
			}
		})
	}
}
