package ncmapi

import (
	//"encoding/json"
	json "github.com/json-iterator/go"
	"io/ioutil"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"
)

// API Paths
const (
	PathLogin          = "/login/cellphone"
	PathTopPlaylists   = "/top/playlist"
	PathPlaylistDetail = "/playlist/detail"
	PathPlaylistTracks = "/playlist/track/all"
	PathHotComment     = "/comment/hot"
	PathLyric          = "/lyric"
)

type Client interface {
	login() (*LoginResult, error)
	TopPlaylists(limit, offset int) (*TopPlaylistsResult, error)
	TopPlaylistsCatalog(catalog string, limit, offset int) (*TopPlaylistsResult, error)
	PlaylistDetail(id ID) (*PlaylistDetailResult, error)
	PlaylistTracks(id ID, limit, offset int) (*PlaylistTracksResult, error)
	TrackHotComment(id ID) (*HotCommentsResult, error)
	Lyric(id ID) (*LyricResult, error)
}

type client struct {
	Phone       string
	PasswordMD5 string
	Server      string
	Cookies     string
	HttpClient  http.Client
}

func newClient(phone string, passwordMD5 string, server string) (*client, error) {
	// 用 jar 维护 cookies
	jar, _ := cookiejar.New(nil)
	httpClient := http.Client{Jar: jar}

	c := &client{
		Phone:       phone,
		PasswordMD5: passwordMD5,
		Server:      server,
		HttpClient:  httpClient,
	}

	// 有 CookieJar 似乎就不用手动维护 Cookies 了，所以这里 LoginResult 没用。
	var err error
	for i := 0; i < NewClientMaxRetryTimes; i++ {
		_, err = c.login()
		if err == nil {
			continue
		}
		logger.Error("failed to create new client, try again after " + strconv.Itoa(NewClientRetryAfterSeconds) + " seconds...")
		time.Sleep(time.Duration(NewClientRetryAfterSeconds) * time.Second)
	}

	return c, err
}

// 用 ClientPool，不能 New
//func NewClient(phone string, passwordMD5 string, server string) Client {
//	return newClient(phone, passwordMD5, server)
//}

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
func (c *client) parseResp(resp *http.Response, result interface{}) (body []byte, err error) {
	defer resp.Body.Close()
	body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		return body, err
	}
	if resp.StatusCode != http.StatusOK {
		return body, NetworkError(resp.Status)
	}
	return body, json.Unmarshal(body, result)
}

// requestAPI = get | post + parseResp + Result.success
func (c *client) requestAPI(req method, apiUrl string, params url.Values, result interface{}) error {
	resp, err := req(apiUrl, params)
	if err != nil {
		return err
	}

	body, err := c.parseResp(resp, result)
	if err != nil {
		return err
	}

	switch r := result.(type) {
	case Resulter:
		if r.success() {
			return nil
		}
	}
	err = NcmError("resp.(Resulter).success failed: " + string(body))
	return err
}

// url.Join(c.Server, subpath)
func (c *client) apiUrl(subpath string) string {
	u, _ := url.Parse(c.Server)
	u.Path = path.Join(u.Path, subpath)
	return u.String()
}

// endregion API request helpers

// region APIs

func (c *client) login() (*LoginResult, error) {
	apiUrl := c.apiUrl(PathLogin)

	params := url.Values{}
	params.Set("phone", c.Phone)
	params.Set("md5_password", c.PasswordMD5)

	result := LoginResult{}

	err := c.requestAPI(c.get, apiUrl, params, &result)
	return &result, err
}

func (c *client) TopPlaylists(limit, offset int) (*TopPlaylistsResult, error) {
	return c.TopPlaylistsCatalog("", limit, offset)
}

// TopPlaylistsCatalog 就是带分类参数的 TopPlaylists。
// 分类列表可以看 AllTopPlaylistsCatalogs
func (c *client) TopPlaylistsCatalog(catalog string, limit, offset int) (*TopPlaylistsResult, error) {
	apiUrl := c.apiUrl(PathTopPlaylists)

	params := url.Values{}
	params.Set("limit", strconv.Itoa(limit))
	params.Set("offset", strconv.Itoa(offset))

	catalog = strings.TrimSpace(catalog)
	if len(catalog) != 0 {
		params.Set("cat", catalog)
	}

	result := TopPlaylistsResult{}

	err := c.requestAPI(c.get, apiUrl, params, &result)
	return &result, err
}

func (c *client) PlaylistDetail(id ID) (*PlaylistDetailResult, error) {
	apiUrl := c.apiUrl(PathPlaylistDetail)

	params := url.Values{}
	params.Set("id", strconv.FormatInt(id, 10))

	result := PlaylistDetailResult{}

	err := c.requestAPI(c.get, apiUrl, params, &result)
	return &result, err
}

func (c *client) PlaylistTracks(id ID, limit, offset int) (*PlaylistTracksResult, error) {
	apiUrl := c.apiUrl(PathPlaylistTracks)

	params := url.Values{}
	params.Set("id", strconv.FormatInt(id, 10))
	params.Set("limit", strconv.Itoa(limit))
	params.Set("offset", strconv.Itoa(offset))

	result := PlaylistTracksResult{}

	err := c.requestAPI(c.get, apiUrl, params, &result)
	return &result, err
}

func (c *client) TrackHotComment(id ID) (*HotCommentsResult, error) {
	apiUrl := c.apiUrl(PathHotComment)

	params := url.Values{}
	params.Set("id", strconv.FormatInt(id, 10))
	params.Set("type", "0") // const type=0 means "comment of track"

	result := HotCommentsResult{}

	err := c.requestAPI(c.get, apiUrl, params, &result)
	return &result, err
}

func (c *client) Lyric(id ID) (*LyricResult, error) {
	apiUrl := c.apiUrl(PathLyric)

	params := url.Values{}
	params.Set("id", strconv.FormatInt(id, 10))

	result := LyricResult{}

	err := c.requestAPI(c.get, apiUrl, params, &result)
	return &result, err
}

// endregion APIs

// region Errors

type NetworkError string

func (n NetworkError) Error() string { return string(n) }

type NcmError string

func (n NcmError) Error() string { return string(n) }

// endregion Errors

// AllTopPlaylistsCatalogs 歌单分类列表。
// 来自 https://music.163.com/#/discover/playlist -> (左上)全部分类
var AllTopPlaylistsCatalogs = []string{
	// 全部风格（总榜）
	"",
	// 语种
	"华语", "欧美", "日语", "韩语", "粤语",
	// 风格
	"流行", "摇滚", "民谣", "电子", "舞曲", "说唱", "轻音乐", "爵士", "乡村",
	"R&B/Soul", "古典", "民族", "英伦", "金属", "蓝调", "雷鬼", "世界音乐",
	"拉丁", "New Age", "古风", "Bossa Nova",
	// 场景
	"清晨", "夜晚", "学习", "工作", "午休", "下午茶", "地铁", "驾车", "运动",
	"旅行", "散步", "酒吧",
	// 情感
	"怀旧", "清新", "浪漫", "伤感", "治愈", "放松", "孤独", "感动", "兴奋",
	"快乐", "安静", "思念",
	// 主题
	"综艺", "影视原声", "ACG", "儿童", "校园", "游戏", "70后", "80后", "90后",
	"网络歌曲", "KTV", "经典", "翻唱", "吉他", "钢琴", "器乐", "榜单", "00后",
}
