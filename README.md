# LandSurfaceClustering

Landon Halloran  /  www.ljsh.ca  /  03.2019

10.2019: If you use this code or some form of it in published work, please cite this repository:
`@misc{LandSurfaceClustering,
  author = {Halloran, L.J.S.},
  title = {LandSurfaceClustering},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lhalloran/LandSurfaceClustering}}
}`

08.2019: If you are interested on collaborating to do something interesting with this type of analysis...send me an email.

This is a script that reads in remote sensing data, performs k-means clustering on sample pixels from the images, and then plots the result. As this is an unsupervised learning algorithm, some knowledge of the "ground truth" will be needed in order to interpret results.
The script might be made more general in the future... for now, you will need to edit it manually.



# Example data
9 bands of Sentinel-2 data in 8-bit png format, some bands resampled to 10m resolution.

# Example output:
![Example output. 8 clusters. Seeland, Neuchatel and Bern Cantons, Switzerland.](extras/output_example_8clusters_Seeland_Switzerland_map.png)
![Example output. 8 clusters. Seeland, Neuchatel and Bern Cantons, Switzerland.](extras/output_example_8clusters_Seeland_Switzerland_pairplot.png)
