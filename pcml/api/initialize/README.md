## Initialize API

In the future, requests to initialize analyses can be made through this API.

### Deployment

The function can be deployed with the following:

```bash

python -m antidote.api.initialize.deploy

```

### Requests

```bash

curl https://{your zone}-{your project}.cloudfunctions.net/initialize -H "Authorization: bearer $(gcloud auth print-identity-token)"

```
