import sys


subscription_key = "d5441656363f4dfdba8d2d7d5ebb95e0"
search_url = "https://api.bing.microsoft.com/v7.0/search"
search_term = sys.argv[1]
import requests

headers = {"Ocp-Apim-Subscription-Key": subscription_key}
params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
search_results = response.json()
from IPython.display import HTML

rows = "\n".join(["""<tr>
                       <td><a href=\"{0}\">{1}</a></td>
                       <td>{2}</td>
                     </tr>""".format(v["url"], v["name"], v["snippet"])
                  for v in search_results["webPages"]["value"]])
HTML("<table>{0}</table>".format(rows))

print(rows)

for v in search_results["webPages"]["value"]:
    url=v["url"]
    request.get(url)
