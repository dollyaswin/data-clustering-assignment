# Dresses_Attribute_Sales

This dataset contain Attributes of dresses and their recommendations according to their sales. Sales are monitor on the basis of alternate days.

## Dataset Information

### Associated Tasks
Classification, Clustering

### #Instances
501

### #Features
13

### Additional Information
Style,	Price,	Rating,	Size,	Season,	NeckLine,	SleeveLength,	waiseline,	Material,	FabricType,	Decoration,	Pattern, Type,	Recommendation are Attributes in dataset.

### Variable Information
- **Style:** `Bohemia,brief,casual,cute,fashion,flare,novelty,OL,party,sexy,vintage,work`.
- **Price:** `Low,Average,Medium,High,Very-High`
- **Rating:** `1-5`
- **Size:** `S,M,L,XL,Free`	
- **Season:** `Autumn,winter,Spring,Summer`
- **NeckLine:** `O-neck,backless,board-neck,Bowneck,halter,mandarin-collor,open,peterpan-collor,ruffled,scoop,slash-neck,square-collar,sweetheart,turndowncollar,V-neck.`
- **SleeveLength:** `full,half,halfsleeves,butterfly,sleveless,short,threequarter,turndown,null`
- **waiseline:** `dropped,empire,natural,princess,null`.	
- **Material:** `wool,cotton,mix, etc`	
- **FabricType:** `shafoon,dobby,popline,satin,knitted,jersey,flannel,corduroy, etc`
- **Decoration:** `applique,beading,bow,button,cascading,crystal,draped,embroridary,feathers,flowers, etc`
- **Pattern type:** `solid,animal,dot,leapard,etc`
- **Recommendation:** `0,1`

### Source
[Dresses_Attribute_Sales at UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/289/dresses+attribute+sales)

# Clustering

## K-Means 

### K = 2
![K = 2](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/k-means/plot-k2.png?raw=true)

### K = 3
![K = 3](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/k-means/plot-k3.png?raw=true)

### K = 4
![K = 4](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/k-means/plot-k4.png?raw=true)

### K = 5
![K = 4](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/k-means/plot-k5.png?raw=true)

### Elbow Method
![Elbow Method for Optimal K](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/k-means/plot-sse.png?raw=true)

### Clustering Result Using the Optimal K
![Clustering using the Optimal K](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/k-means/plot-k.png?raw=true)

### New Datasheet with clustering data 
![New Datasheet Update](https://github.com/dollyaswin/data-clustering-assignment/blob/main/assets/dress-datasheet-k-means-clustering.png?raw=true)

## Agglomerative

### Create Dendogram
![Creating Endogram](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/agglomerative/dendogram.png?raw=true)

### Create Silhoutte Scoring
![Creating Silhouette Scoring](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/agglomerative/plot-silhouette.png?raw=true)

### Clustering Result Using the Optimal K
![Clustering using the Optimal K](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/agglomerative/plot-k.png?raw=true)

### New Datasheet with clustering data 
![New Datasheet Update](https://github.com/dollyaswin/data-clustering-assignment/blob/main/assets/dress-datasheet-k-means-clustering-screen.png?raw=true)
