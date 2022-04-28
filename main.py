import pandas as pd
from libtlda.flda import FeatureLevelDomainAdaptiveClassifier


def run():
    src = pd.read_csv('./source_data_feature.csv')
    src_label = pd.read_csv('./source_data_label.csv')
    dst = pd.read_csv('./dst_data_feature.csv')

    cls = FeatureLevelDomainAdaptiveClassifier()
    cls.fit(src,src_label,dst)
    
    



if __name__ == '__main__':
    run()