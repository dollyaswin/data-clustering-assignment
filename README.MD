# Dresses_Attribute_Sales

This dataset contain Attributes of dresses and their recommendations according to their sales. Sales are monitor on the basis of alternate days.

## Dataset Information

### Associated Tasks
Classification, Clustering

### #Instances
501

### #Features
13

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

### Datasheet Preview 
![Datasheet Preview](https://github.com/dollyaswin/data-clustering-assignment/blob/main/assets/dress-datasheet-preview.png?raw=true)

### Source
[Dresses_Attribute_Sales at UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/289/dresses+attribute+sales)

# Clustering

## K-Means (Price vs Rating) 

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

### Silhouette Scoring
![Creating Silhouette Scoring](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/k-means/plot-silhouette.png?raw=true)

### Clusters Using Optimal K
![Clustering using the Optimal K](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/k-means/plot-k.png?raw=true)

#### Clusters
| **No** | **Price**        | **Rating** |
|--------|------------------|------------|
|   1.   | Average - Medium |    `3 - 5`   |
|   2.   | Low              |    `4 - 5`   |
|   3.   | Low - Medium     |    `0 - 1`   |
|   4.   | High - Very High |    `4 - 5`   |
|   5.   | High - Very High |      `0`     |

### New Datasheet Preview With Cluster 
![New Datasheet Update](https://github.com/dollyaswin/data-clustering-assignment/blob/main/assets/dress-datasheet-k-means-clustering.png?raw=true)

## Agglomerative (Price vs Rating)

### Dendrogram
![Creating Dendrogram](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/agglomerative/dendogram.png?raw=true)

### Silhouette Scoring
![Creating Silhouette Scoring](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/agglomerative/plot-silhouette.png?raw=true)

### Clustering Result Using the Optimal K
![Clustering using the Optimal K](https://github.com/dollyaswin/data-clustering-assignment/blob/main/visualizations/agglomerative/plot-k.png?raw=true)

#### Clusters
| **No** | **Price**        | **Rating** |
|--------|------------------|------------|
|   1.   | Average          |   `3 - 5`  |
|   2.   | Low              |   `4 - 5`  |
|   3.   | Low              |   `0 - 1`  |
|   4.   | High - Very High |  `4.3 - 5` |
|   5.   | Very High        |  `4.1 - 5` |
|   6.   | Average          |     `0`    |
|   7.   | Medium           |  `4.2 - 5` |
|   8.   | Medium           |     `0`    |
|   9.   | High             |     `0`    |
|   10.  | High - Very High |     `0`    |

### New Datasheet Preview With Cluster 
![New Datasheet Update](https://github.com/dollyaswin/data-clustering-assignment/blob/main/assets/dress-datasheet-agglomerative-clustering.png?raw=true)
