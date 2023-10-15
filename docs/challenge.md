
## LATAM ML Engineer Challenge Docs

### Practices applied

- `pylint` used as linter and static code analyzer, plus `pylance` vscode extension.
-` black` as code formatter.
- Live version deployed to ** GCP Cloud Run** service.

**Observations:** Due to Challenge constraints, files such as .gitignore, .dockerignore, coverage folder, and reports folders were omitted from commits.
### Some key takeways

#### Python version selected
- Due to the latest docker Python image is 3.12, some libraries could have issues, so I decided to go for a well-tested image
such as Python version 3.11.4, where all the libraries used on the challenge are compatible.

#### Model pre-loading on API startup
- When the model is trained, in addition to being saved on the internal attribute `self._model`, it's saved as a **pickle file** on the
root directory of the project, with this implementation we can load the model on API startup and only compromise
the startup time of the API, then each time a request is received the model is already loaded in memory, that way the API
avoid re-train the model with the same dataset each time a new request arrives.

- On a real productive component, is expected that the training process is executed outside the proper API component, and the model is just
downloaded/copied or mounted from an external storage system such as a blob storage system, an NFS, etc. Also, is expected a proper versioning
 system for the model.

#### Extra method on DelayModel class
- Some classes related to derived features ( "period_day", "high_season", "min_diff") that are not considered at the end of the process, 
are still implemented, that's because if the model needs to use a derived feature as a target we can re-utilize the same class, just changing the inputs.

### Model Selection
- As the notebook shows both models perform similarly both classification reports show similar metrics for precision, recall, and f1-score,
 in terms of metrics any could be selected.
- Now, in terms of implementation, time constraints, package dependencies, and ease of use, I just selected `LogisticRegression`.
That means one less 3rd-party dependency for the component, and the straightforwardness of the library helps to deliver a complete solution in time. 

### Fixes applied to make it works
- Updated the numpy library due to incompatibility of the compiled version with the Python image used. First I used the following:
  ```
  pip install numpy --upgrade
  ```
  Then the version was updated on `requirements.txt` file

- Seaborn library needs to use the keywords of the arguments passed on a method, so in the Notebooks, I update all the barplot methods. 
The notebook works as expected.
- Some hint typing definitions were malformed or incompleted, but most of them were fixed.

When running the unit test via make file, some imports or incompatibility issues appears, to make it work properly, the following changes
were performed:

- FastAPI bumped due to a test import error related to TestClient class, starlette, httpx, and anyio dependency libraries.
```
fastapi test client AttributeError: module 'anyio' has no attribute 'start_blocking_portal'
```
- Locust version was bumped, similar error, in this case httpx installed version was no compatible with Locust version.

### How to run it locally

#### On local machine

Create a virtualenv with Python version 3.11.4 [Virtualenv](https://virtualenv.pypa.io/en/latest/)

All the following commands are executed from the root dir of the repository:

Install Python dependencies
```
pip install --upgrade pip
pip install -r requirements.txt

# If you want to run the Jupyter notebook
pip install -r requirements.txt

# If you want to run test
pip install -r requirements-test.txt
```

Run API
```
uvicorn challenge.api:app --reload
```

#### Docker

Build Dockerfile
```
docker build -t delay_api . 
```

Run container
```
docker run -p 80:80 delay_api
```
### Test it live

You can test it on the following url (it will be shutdown after review) https://latam-challenge-bt4bd7swda-ue.a.run.app
