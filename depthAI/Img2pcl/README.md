### Install Dependencies:
`python3 install_requirements.py`

Note: `python3 install_requirements.py` also tries to install libs from requirements-optional.txt which are optional. For ex: it contains open3d lib which is necessary for point cloud visualization. However, this library's binaries are not available for some hosts like raspberry pi and jetson.   

### Running Example As-Is:
`python3 main.py` - Runs without point cloud visualization
`python3 main.py -pcl` - Enables point cloud visualization


This will run subpixel disparity by default.
