// Package ncmapi offers NeteaseCloudMusicApi clients.
// Call GetClient() to get a client, after Init() of course:
//     cli := GetClient()
// Then use APIs:
//     result, err := cli.TopPlaylists(50, 0)
// And put the client back after using:
//     Done(cli)
package ncmapi
