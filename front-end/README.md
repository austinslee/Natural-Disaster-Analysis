# So i will just cover major components in this read me

This website is builded using javascript and framework is react
You would also need to make sure that javascript, node js, IDE(visual studio code or any other IDE) is installed in your computer to run this website
To start this wesbsite go to terminal and pass command: npm start

If you want to make any changes in the website then go to :
For (First page: This is where you type the search query)home page: Go to src -> Components -> Home
For (second page: This is where results are shown for your typed query and also you can pass filter query) results page: Go to src -> Components -> results
For (Visualization page: This is where map is shown with data points) Visualization page: Go to src -> Components -> visualization
For (Disaster Details page: This page is shown when i click on a particular disaster and details are shown) Result details page: Go to src -> components -> result_details

So .js files contain the logic code and layout code (website template) and .css files contains component design properties.

Also you would have to create a account on https://kepler.gl/ website to get the mapboxApiAccessToken. To make the load the map on the website.

To store api results i have also used redux in this website. To change anything in redux: Go to src -> redux
So, basically redux is used to manage state and store values globally. 
In src -> images -> disasterImages i have stored all the disaster type images that i show when displaying the results.

For other things i have included comments in code at major places for you to understand what my codes does.
