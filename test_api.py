import requests

url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
params = {
    'parameters' : 'T2M,PRECTOTCORR',
    'community'  : 'RE',
    'longitude'  : -90.88,
    'latitude'   : 14.47,
    'start'      : 2015,
    'end'        : 2023,
    'format'     : 'JSON'
}
response = requests.get(url, params=params, timeout=30)
print(response.status_code)  
data = response.json()
print(list(data['properties']['parameter']['T2M'].items())[:3])