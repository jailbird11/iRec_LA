# i-Recommend
Mining Association rules in student curriculum and Recommending courses to the students based on their interests.

## Project Architecture
## Libraries or Algorithms Used
For Identifying the frequently taken Courses, Association Rule Mining is used and for recommending courses to the Students,Collaborative Filtering is used.Following are the libraries used:
Visualization and Frontend- Dash Plotly and Dash Bootstrap
Association Rule Mining - mlXtend
Collaborative Filtering - Surprise (KNNwithMeans)
## Screenshots
![alt text](https://github.com/swarnas89/LAProject1/blob/master/rec1.png)
![alt text](https://github.com/swarnas89/LAProject1/blob/master/rec2.png)
## Running and Deploying the Solution
Use Jupyter Notebook and click Run all,then the application is deployed automatically. Instead,if we need to deploy on a different server,enter the port number and server to be deployed as parameters for the method indicated below:
if __name__ == '__main__':
    app.run_server(debug=True) -->here
