## Feature Subest Selection Toolkit

### Setup ###

* Cloning the repository
```
git clone git@bitbucket.org:gsudmlab/fss_toolkit.git
```

* Installing dependencies
```
cd fss_toolkit
python -m venv venv
source ./venv/bin/activate
(venv) fss_toolkit> pip3 install -r ./requirements.txt
```

* Demo Code - demo.ipynb

### Project Structure

```
mvts_fss_scs
    |__ images
    |__ mvts_fss_scs
            |__ data
                    |__ partition 1
                        |__ FL
                        |__ NF
                    |__ partition 2
                        |__ FL
                        |__ NF
            |__ __init__.py
            |__ evaluation
                    |__ __init__.py
                    |__ metric.py
                    |__ plotter.py
                    |__ multivariate.py
                    |__ univariate.py
            |__ fss
                    |__ __init__.py
                    |__ base_fss.py
                    |__ clever
                    |__ corona
                    |__ csfs
                    |__ fcbf
                    |__ pie
                    |__ mrmr_relief
                    |__ rfe
            |__ preprocessing
                    |__ __init__.py
                    |__ imputer.py
                    |__ labeler.py
                    |__ normalizer.py
                    |__ sampler.py
                    |__ vectorizer.py

    |__ Results
    |__ CONSTANTS.py
    |__ README.md
    |__ requirements.txt
```

### Acknowledgement
This work was supported in part by two NASA Grant Awards [No. NNH14ZDA001N, 80NSSC20K1352], and two NSF Grant Awards [No. AC1443061 and AC1931555]. The AC1443061 award has been supported by funding from the Division of Advanced Cyber infrastructure within the Directorate for Computer and Information Science and Engineering, the Division of Astronomical Sciences within the Directorate for Mathematical and Physical Sciences, and the Division of Atmospheric and Geospace Sciences within the Directorate for Geosciences.

### License
This software is distributed using the [GNU General Public License, Version 3](./LICENSE.txt)  
 ![alt text](./images/gplv3-88x31.png)