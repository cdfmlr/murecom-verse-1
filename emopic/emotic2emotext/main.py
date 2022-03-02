import json
import sys

from emotic2emotext import emotic2emotext

if __name__ == '__main__':
    # emotic/server output json => emotext emotions
    # curl -F "img=@/pathto/image.jpg" http://localhost:8080/infer | python3 emotic2emotext/main.py
    for line in sys.stdin:
        out = []
        for res in json.loads(line):
            dlut = emotic2emotext(res['cat'], res['cont'])
            out.append({'bbox': res['bbox'], 'emotions': dlut})
        sys.stdout.write(json.dumps(out) + '\n')
