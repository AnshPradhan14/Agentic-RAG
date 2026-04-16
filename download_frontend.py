import urllib.request
import os

os.makedirs('frontend', exist_ok=True)

screens = {
    "login": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzE3NWI3ZWI5ZWIxMDQ0ZWFiMmQ2MTg1MjJjMTcxMzliEgsSBxDigOqi5x8YAZIBIwoKcHJvamVjdF9pZBIVQhM5OTE0ODkzOTQ3ODUwMTI5NTI0&filename=&opi=96797242",
    "chat": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sX2VhOThkNTJmYWM2NzQyOTJhNzA5OTMxOTAwYThlM2I4EgsSBxDigOqi5x8YAZIBIwoKcHJvamVjdF9pZBIVQhM5OTE0ODkzOTQ3ODUwMTI5NTI0&filename=&opi=96797242",
    "admin_login": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzJlMWQzNDgzMTNiODQwYTA4MWFiNmQ4ZDdiNDYyZjQ5EgsSBxDigOqi5x8YAZIBIwoKcHJvamVjdF9pZBIVQhM5OTE0ODkzOTQ3ODUwMTI5NTI0&filename=&opi=96797242",
    "admin_dashboard": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sX2FjMGM2ZmU0YzgxNzRiMTA5ZmE3OTQ1YmM5N2RlNzBjEgsSBxDigOqi5x8YAZIBIwoKcHJvamVjdF9pZBIVQhM5OTE0ODkzOTQ3ODUwMTI5NTI0&filename=&opi=96797242",
    "admin_settings": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzgwMmM3OWIxOWEyYzRkOWI4NTEzMDNiOTVmMTQwZTM1EgsSBxDigOqi5x8YAZIBIwoKcHJvamVjdF9pZBIVQhM5OTE0ODkzOTQ3ODUwMTI5NTI0&filename=&opi=96797242"
}

for name, url in screens.items():
    try:
        response = urllib.request.urlopen(url)
        content = response.read().decode('utf-8')
        with open(f'frontend/{name}.html', 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Downloaded {name}.html")
    except Exception as e:
        print(f"Failed to download {name}: {e}")

print("Download complete.")
