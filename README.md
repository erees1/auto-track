# auto-track

Simple python tool to import transactions into an excel spending tracker

Set up configs for the following, i.e. know where to look in e.g. monzo csv transactions export 

* Monzo
* Amazon

Functionality to automatically categorise transactions does not work yet!

## Usage

```bash
python3 -m venv ./venv
pip install -r requirements.txt
```

Options are set in [`config/config.json`](config/config.json)

```bash
# selects env and uses default path to excel output 
auto-track/import.sh <path to file to import>

# Use python interface
python3 src/import.py <path to file to import> <path to excel output> 
```
