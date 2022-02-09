package neteasecloudmusic

import (
	"encoding/json"
	"net/http"
	"net/url"
)

type Client interface {
	login(phone, passwordMD5 string) (LoginResult, error) // post
	TopPlaylists(limit int) (TopPlaylistsResult, error)   // post
	PlaylistDetail()                                      // TODO
}

type client struct {
	Phone       string
	PasswordMD5 string
	Cookies     string
	HttpClient  http.Client
}

// region API request helpers

// method: get | post
type method func(baseUrl string, params url.Values) (resp *http.Response, err error)

func (c *client) get(baseUrl string, params url.Values) (resp *http.Response, err error) {
	req, err := http.NewRequest(http.MethodGet, baseUrl, nil)
	req.URL.RawQuery = params.Encode()
	return c.HttpClient.Do(req)
}

func (c *client) post(baseUrl string, params url.Values) (resp *http.Response, err error) {
	return c.HttpClient.PostForm(baseUrl, params)
}

// *http.Response -> Body -> json -> result
func (c *client) parseResp(resp *http.Response, result interface{}) error {
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return NetworkError(resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(result)
}

// requestAPI = get | post + parseResp
func (c *client) requestAPI(req method, apiUrl string, params url.Values, result interface{}) error {
	resp, err := req(apiUrl, params)
	if err != nil {
		return err
	}
	return c.parseResp(resp, result)
}

// endregion API request helpers

// region APIs

// endregion APIs

// region Errors

type NetworkError string

func (n NetworkError) Error() string {
	return string(n)
}

// endregion Errors
