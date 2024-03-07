## Infer API

Model inference API

### Deployment

The function can be deployed with the following:

```bash

python -m antidote.api.infer.deploy

```

Note: This deployment is comparatively slower than many because the full project requirements.txt is installed when the runtime is built which includes at last checking all of tensorflow and pytorch.

### Requests

```bash

curl https://{your zone}-{your project}.cloudfunctions.net/infer -H "Authorization: bearer $(gcloud auth print-identity-token)"

```
