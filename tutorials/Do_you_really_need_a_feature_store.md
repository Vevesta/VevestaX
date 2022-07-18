# Do you really need a Feature Store ?

![img](https://miro.medium.com/max/788/0*HKy72KjC51qO7Av-)

Photo by [Compare Fibre](https://unsplash.com/@comparefibre?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral)

Feature store’s strength lies predominantly in the fact that it brings data from disparate sources especially, time-stamped clickstream data and provides them to data scientists as and when needed. But a deeper dive reveals a lot of use cases that are dominant in the data science community and are in fact overlooked by feature stores.

## When are Feature Stores useful :
Feature stores are useful since they enable data scientists to compute features on the server, say number of clicks. The alternative to this is computing features in the model itself or/and compute in a transform function in SQL.

## Where Feature Stores fail ?
1. **Steep learning curve:** The code that is required to integrate features with the feature store is not simple for a data scientist from non-programming backgrounds. Data Scientists are in general exposed to Pandas and SQL type of syntax. The learning curve required to work feature stores is by no means small.

2. **Little overlap in features being used:** Feature stores enables you to reuse features. In data science, features require some pre-processing, so either values are imputed in the features or some aggregation of features is done, say value of feature in the last 24 hours or last week. For feature stores to be of real use, not only do these features need to be computationally expensive, they also require reuse of features by multiple data science teams. But, this is generally not the case, each project will use its own data imputation and aggregation, depending on the problem at hand.

3. **Risk of change is pipelines:** Feature stores enable extensive collaboration between data scientists. But the implicit requirement is that data scientists for a particular project might use, say, imputation of mode during pre-processing, but then they later decide to change it to, say, mean. This will require the data scientist to create a new feature in the feature store and change his complete pipeline or change the definition of feature in feature store which in turn would require other data scientists dependent on the feature to change their pipelines, neither is a comfortable option.
In short, feature stores are clearly suited for the narrow use case of creation of features from time series data that are extensively computationally expensive features. Most organizations don’t have this use case in place.

## Credits:
The above article is sponsored by [***Vevesta.***](http://www.vevesta.com/?utm_source=Github_VevestaX_FeatureStore)

[***Vevesta:***](http://www.vevesta.com/?utm_source=Github_VevestaX_FeatureStore) Your Machine Learning Team’s Collective Wiki: Save and Share your features and techniques.

For more such stories, follow us on twitter at [@vevesta1](http://twitter.com/vevesta1).
