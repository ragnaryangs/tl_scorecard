# !/usr/bin/env python
# coding:utf-8

# 这样可以直接from tl_scorecard import 各个函数，否则需要from tl_scorecard import T01_DataAnalysis，T01_DataAnalysis.func(**)
from tl_scorecard.T01_DataAnalysis import *   
from tl_scorecard.T02_FeatureSelection import * 
from tl_scorecard.T03_Binning import * 
from tl_scorecard.T04_ModelBuilding import * 
from tl_scorecard.T05_ModelEvaluation import * 
from tl_scorecard.T06_ScoreCard import * 
from tl_scorecard.T07_Report import * 
from tl_scorecard.T08_CardApply import *