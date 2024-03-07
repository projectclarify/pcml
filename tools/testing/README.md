## Testing and PRs

### Local testing:

Local testing should be performed using minikube which can be installed via instructions [here](https://minikube.sigs.k8s.io/docs/start/).

In order to pull images from Google Container Registry from within minikube, you'll need to authenticate. Two methods are described [here](https://ryaneschinger.com/blog/using-google-container-registry-gcr-with-minikube/). The following works well enough:

```bash
kubectl --namespace=default \
          create secret docker-registry gcr \
          --docker-server=https://gcr.io \
          --docker-username=oauth2accesstoken \
          --docker-password="$(gcloud auth print-access-token)" \
          --docker-email=youremail@example.com
```


```bash
kubectl --namespace=default \
          patch serviceaccount default \
          -p '{"imagePullSecrets": [{"name": "gcr"}]}'
```

During local development you can alternatively use a remote kubernetes deployment but unless nodes are left running, repeated pulling of runtime images (based on ubuntu) will cause a lot of delay with each test. Locally, after the first test, most of the runtime image will be cached on your local machine.

#### Local debug

It is possible that tests will pass on your local but not on our remote test systems. To debug this you can run these tests locally (in the same containers as are run remotely) on a machine with Docker, e.g.

```bash

python tools/environments/build.py --build_mode=local --static_image_id=test-container --container_type=workspace

docker run -it test-container /bin/bash -c "source ~/.bashrc; cd /build; sh tools/testing/test_local.sh"

```

### Remote testing:

When a PR is submitted, CircleCI tests are always run including operations that don’t require special credentials as well as linting and test coverage analysis. 

