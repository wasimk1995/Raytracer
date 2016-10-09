##Iterative Raytracer Algorithm

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Implemented inherintly recursive raytracer algorithm on GPU using iterative method in Cuda. Each pixel was mapped with a single ray that interacted with the background image and produced the color for different depths.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Using the iterative method, I calculated the all combinations of reflected and refracted rays for n = 5 depth and worked backwards to find contribution of each ray to the original pixel. With depth 5, each ray would spawn result in 63 total rays. Depending on the resolution, I was able to parallelize each pixel on the GPU.