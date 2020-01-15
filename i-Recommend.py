#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#StudentCourseData_Report10.csv -- Dataset


# In[661]:


import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash.dependencies
from dash.dependencies import Input, Output, State
import dash_daq as daq
import plotly
import plotly.graph_objs as go
import networkx as nx
import plotly.express as px
import dash_cytoscape as cyto
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from surprise import SVD, accuracy,Dataset
from surprise.reader import Reader
from surprise.model_selection import cross_validate,train_test_split
from surprise import SVD,SVDpp,SlopeOne,NMF,NormalPredictor,KNNBaseline,KNNBasic,KNNWithMeans,KNNWithZScore,BaselineOnly,CoClustering


# In[182]:


df=pd.read_csv('StudentCourseData_Report10.csv')


# In[187]:


df.to_csv('studdata1.csv',header=False, index=False)


# In[188]:


df_t=pd.read_csv('studdata1.csv')


# In[189]:


df_t.head(3)


# In[190]:


df_t.drop(['Unnamed: 31','Unnamed: 32','Unnamed: 33'],axis=1,inplace=True)


# In[191]:


df_t.index.names=['id']


# In[192]:


df_t = df_t.astype(object)


# In[193]:


df_t.reset_index(level=0,inplace=True)


# In[194]:


df_t.reset_index(inplace=False)


# In[195]:


df_test= df_t.melt(id_vars="id",var_name="sub")


# In[200]:


df_test.head()


# In[201]:


df2= df_test["value"].str.extractall(r":(\d+|\w)").unstack()


# In[202]:


df2.head(2)


# In[203]:


df2.columns=["semester","grade","rating"] 


# In[205]:


dfrslt= pd.concat([df_test.drop(columns="value"),df2],axis=1)         .reindex(["id","semester","sub","grade","rating"],axis=1)         .sort_values("id")


# In[213]:


dfrslt.drop(dfrslt[dfrslt['semester']=='D'].index,axis=0,inplace=True)


# In[214]:


dfrslt[dfrslt['semester']=='D']


# In[216]:


dfrslt_nan=dfrslt.dropna(thresh=3, axis=0)


# In[217]:


df_subs1=dfrslt_nan.drop(['semester','grade','rating'],axis=1)


# In[218]:


df_final=dfrslt_nan.assign(count=dfrslt_nan.groupby(['id','semester']).cumcount()).pivot_table(index = ['id','count'],columns = 'semester',
                        values = 'sub',aggfunc = ''.join).rename_axis(columns = None,index = [None,None])


# In[219]:


df_final.head(40)


# In[296]:


df_final.to_csv('category1.csv')


# In[221]:


df_processed=df_subs1.groupby('id').sub.apply(list).reset_index()


# In[222]:


df_processed.head(5)


# In[223]:


df_processed.set_index('id',inplace=True)


# In[224]:


dataset=list(df_processed['sub'].values)


# In[225]:


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
df_encoded


# In[226]:


from mlxtend.frequent_patterns import apriori

apriori(df_encoded, min_support=0.45,use_colnames=True)


# In[227]:


frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True,verbose=1)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


# In[228]:


from mlxtend.frequent_patterns import association_rules

rules=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.1)


# In[229]:


rules.head()


# In[230]:


len(rules.values)


# In[231]:


l=[]
for i,j in zip(range(0,len(rules.iloc[:,0])),range(0,len(rules.iloc[:,1]))):
    itemset = set(rules.iloc[i,0]) | set(rules.iloc[i,1])
    l.append(tuple(itemset))


# In[232]:


len(l)


# In[233]:


import functools
def reduce_concat(x, sep=""):
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)

def paste(*lists, sep=" ", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)


# In[234]:


antecedents=[]
for i in range(0,len(rules.iloc[:,0])):
    antecedents.append(list(rules.iloc[i,0]))


# In[235]:


consequents=[]
for i in range(0,len(rules.iloc[:,1])):
    consequents.append(list(rules.iloc[i,1]))


# In[236]:


list_join=paste(antecedents, consequents, sep=',')


# In[237]:


df_join=pd.DataFrame(list_join)


# In[238]:


halfjoins=df_join[0].str.replace('[','')


# In[239]:


ohnequote=halfjoins.str.replace(']','')


# In[240]:


finalentry=ohnequote.str.replace("'","")


# In[241]:


newrules=rules.assign(itemsets=finalentry)


# In[242]:


newrules.head(2)


# In[243]:


l1=[]
for val in newrules['itemsets']:
    l1.append((val,))


# In[244]:


newrules_up=newrules.assign(itemsets1=l1,inplace=True)


# In[245]:


newrules_up.drop(['itemsets'],axis=1,inplace=True)


# In[246]:


newrules_up.head(2)


# In[247]:


df_rec=dfrslt_nan


# In[248]:


s=df_rec['sub'].astype('category')


# In[364]:


grade=df_rec['grade'].astype('category')


# In[311]:


df_rec=df_rec.fillna(0)


# In[312]:


df_rec['rating']=df_rec['rating'].astype(int)


# In[249]:


df_rec['course id']=s.cat.codes


# In[365]:


df_rec['grade_codes']=grade.cat.codes


# In[367]:


df_rec[['grade','grade_codes']]


# In[318]:


df_useritem=df_rec[['id','course id','sub','rating']]


# In[319]:


df_useritem.dtypes


# In[317]:


#df_useritem[['rating']]=df_useritem[['rating']].apply(pd.to_numeric) 


# In[252]:


#df_rec[['rating']]=df_rec[['rating']].apply(pd.to_numeric) 


# ## Recommendation System

# In[320]:


reader=Reader(rating_scale=(1, 5))


# In[330]:


data = Dataset.load_from_df(df_useritem[['id', 'course id', 'rating']],reader)


# In[322]:


df_subid=df_useritem[['sub','course id']]


# In[331]:


df_subid.drop_duplicates()


# In[332]:


df_subid.sort_values(by=['course id'], inplace=True)


# In[333]:


df_subid.drop_duplicates(inplace=True)


# In[334]:


df_subid.head(5)


# In[335]:


print('Using ALS')
bsl_options = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }
algo = KNNBaseline(bsl_options=bsl_options)
cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)


# In[343]:


trainset, testset = train_test_split(data, test_size=0.25)
algo = KNNWithMeans()
predictions = algo.fit(trainset).test(testset)
accuracy.rmse(predictions)


# In[341]:


benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
    # Perform cross validation
    try:
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=2, verbose=True)
    except ZeroDivisionError:
        print('Nan values are eliminated')
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')


# In[344]:


list_subs=['Interaktive Systeme','Information Mining','Advanced Web Technologies']


# In[345]:


list_neighbors=algo.get_neighbors(13,k=3)


# In[346]:


list_neighbors


# In[613]:


def get_neighbors(list1,k):
    list_neighbors=[]
    list_subs_up=[]
    for val in list1:
        list_subs_up.append(int(df_subid[df_subid['sub']==val]['course id'].values))  
    for val in list_subs_up:
        list_neighbors.append(algo.get_neighbors(val,k))
        print("The students who chose these subjects chose the following subjects:",algo.get_neighbors(val,k))
    return list_neighbors


# In[614]:


def get_subs(list1):
    l_neigh=[]
    for val in list1:
        l_neigh.append(((df_useritem[df_useritem['course id']==val]['sub']).drop_duplicates()).tolist())
        list_neigh = [item for sublist in l_neigh for item in sublist]
    return list_neigh


# In[349]:


l_up=get_neighbors(list_subs,3)


# In[615]:


def recommend_courses(list_subs):
    recommendation_list=[]
    for val in get_neighbors(list_subs,3):
        recommendation_list.append(get_subs(val))
    return recommendation_list


# In[616]:


recommend_courses(['Advanced Web Technologies','Advanced Image Synthesis','Cognitive Robot Systems','Computer Robot Vision'])


# In[351]:


recommendation_list


# In[352]:


df_sankey_daig=pd.read_csv('category1.csv')


# In[353]:


df_sankey_daig.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[553]:


df_ratings=df_rec.groupby('sub').rating.mean().reset_index()


# In[563]:


df_ratings.head(4)


# In[562]:


df_rec.groupby('sub')['grade_codes'].value_counts()


# ## Dash

# In[354]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# In[355]:


fig3 = px.scatter(newrules_up, x="support", y="confidence", size="lift",hover_name="itemsets1")


# In[356]:


fig5=px.histogram(df_subs1,x='sub')


# In[357]:


fig6=px.scatter(df_rec, x='course id', y='grade',color='sub',size='rating',animation_frame='rating',animation_group='course id')
#px.scatter(df_rec, x="sub", y="rating", animation_frame="rating", animation_group="course id",size='rating',
            #color="course id", hover_name="sub")
#fig6.update_layout(title="")
fig6.show()


# In[361]:


fig7 = px.parallel_categories(df_sankey_daig[20:30], dimensions=['Unnamed: 1','1', '2', '3','4','O'],
                color="Unnamed: 1", color_continuous_scale=px.colors.sequential.Inferno,
                labels={'Unnamed: 1':'Studentpath','1':'Semester1', '2':'Semester2', '3':'Semester3','4':'Semester4','O':'OtherSemesters'},title='Student Behavioral Flow from Semester to Semester',height=500,width=1000)


# In[362]:


fig7.show()


# In[377]:


fig8 = go.Figure(data=[
    go.Bar(name='Ratings', x=list(df_rec['sub'].values), y=list(df_rec['rating'].values)),
    go.Bar(name='Semester', x=list(df_rec['sub'].values), y=list(df_rec['semester'].values)),
    go.Bar(name='Grading', x=list(df_rec['sub'].values), y=list(df_rec['grade_codes'].values),hovertext=list(df_rec['grade'].values))])


# In[382]:


fig8.update_xaxes(tickfont=dict(size=7))


# In[712]:


app = dash.Dash(__name__,external_stylesheets=external_stylesheets)


# In[713]:


colors = {
    'background': '#6f32a8',
    'text': '#f2f542',
    'background-image': 'url("https://www.e-spincorp.com/wp-content/uploads/2019/03/Augmented-analytics-Capabilities-in-Business-Intelligence-1024x688.jpg")'
}


# In[714]:


app.layout = html.Div(style={'background-image': colors['background-image']},children=[
    html.H1('Student Course Visualization', style={
            'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'georgia','color':'#3269a8','background-color': 'lightblue'}),
    dcc.Tabs(id="tabs", vertical=False,children=[
        dcc.Tab(label='Home', children=[
            html.H2('Welcome to the Course Advisory System',style={'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'georgia','color':'#3269a8'}),
            html.H4('Select the List of Courses you would like to attend:',style={'textAlign': 'left', 'margin': '48px 0', 'fontFamily': 'georgia','color':'#3269b8'}),
            html.Div(dcc.Input(id='input1',value='Enter keyword',type='hidden')),
            html.Div([dcc.Checklist(id='checklist1',
    options=[
        {'label': 'Advanced Image Synthesis', 'value': 'Advanced Image Synthesis'},
        {'label': 'Advanced Web Technologies', 'value': 'Advanced Web Technologies'},
        {'label': 'Cloud Web & Mobile', 'value': 'Cloud Web & Mobile'},
        {'label': 'Cognitive Robot Systems', 'value': 'Cognitive Robot Systems'},
        {'label': 'Computer Robot Vision', 'value': 'Computer Robot Vision'},
        {'label': 'Computer Graphics', 'value': 'Computer Graphics'},
        {'label': 'Gestaltung interaktiver Lehr-/Lern-Systeme', 'value': 'Gestaltung interaktiver Lehr-/Lern-Systeme'},
        {'label': 'Development of Safe and Secure Software', 'value': 'Development of Safe and Secure Software'},
        {'label': 'Digital Games Research', 'value': 'Digital Games Research'},
        {'label': 'Distributed Systems', 'value': 'Distributed Systems'},
        {'label': 'Electronic Communities and Social Networks', 'value': 'Electronic Communities and Social Networks'},
        {'label': 'Fault Diagnosis and Fault Tolerance in Technical Systems', 'value': 'Fault Diagnosis and Fault Tolerance in Technical Systems'},
        {'label': 'Formal Specification of Software Systems', 'value': 'Formal Specification of Software Systems'},
        {'label': 'Game Architecture and Design', 'value': 'Game Architecture and Design'},
        {'label': 'Information Engineering', 'value': 'Information Engineering'},
        {'label': 'Information Mining', 'value': 'Information Mining'},
        {'label': 'Information Retrieval', 'value': 'Information Retrieval'},
        {'label': 'Interaktive Systeme', 'value': 'Interaktive Systeme'},
        {'label': 'Internet of Things- Protocols and System Software', 'value': 'Internet of Things- Protocols and System Software'},
        {'label': 'Knowledge-Based Systems', 'value': 'Knowledge-Based Systems'},
        {'label': 'Learning Analytics', 'value': 'Learning Analytics'},
        {'label': 'Master Seminar Informatics', 'value': 'Master Seminar Informatics'},
        {'label': 'Modelling of Concurrent Systems', 'value': 'Modelling of Concurrent Systems'},
        {'label': 'Modelling-Analysis-Verification', 'value': 'Modelling-Analysis-Verification'},
        {'label': 'Natural-Language-based Human-Computer Interaction', 'value': 'Natural-Language-based Human-Computer Interaction'},
        {'label': 'Neurocomputing and Organic Computing', 'value': 'Neurocomputing and Organic Computing'},
        {'label': 'Pattern and Component based Software Development', 'value': 'Pattern and Component based Software Development'},
        {'label': 'Peer-to-Peer Systems', 'value': 'Peer-to-Peer Systems'},
        {'label': 'Recommender Systems', 'value': 'Recommender Systems'},
        {'label': 'Scientific Visualization', 'value': 'Scientific Visualization'},
        {'label': 'Test and Reliability of Digital Systems', 'value': 'Test and Reliability of Digital Systems'},
        
    ],
                                    #value=[]
  )
                     ],),
            html.Button('Suggest ME!', id='button1',style={'color':'#4287f5','size':'large'}),
            
          #dt.DataTable(id='Table',columns=['Course','Suggested Courses'])
        ]),
        dcc.Tab(label='Graph1', children=[
            html.Div([
                html.H1("This is the content in tab 2"),
                dcc.Dropdown(id='dropdown_courses',
                options=[
                     {'label': 'Rating', 'value': 'Rating'},
        {'label': 'Grade', 'value': 'Grade'},
        {'label': 'Semester', 'value': 'Semester'}
                ]),
                html.Div(id='barchart',)
                
                #html.Div(className='card',children=[html.Img(src='unilogo.png',style={"width:100%"}),
                                                    #html.Div(children=[html.H4("Img1"),html.P("This is image1")])])
            ])
        ]),
        dcc.Tab(label='Graph2', children=[
            html.Div([
                html.H1("Generated Recommendations based on the Courses chosen by you"),
                dcc.Store(id="value", storage_type="memory"),
                dcc.Graph(id='content1'),
            ])
        ]),
    ],
        style={
        'fontFamily': 'system-ui',
        #'text': '#f2f542',
        'background': '#ebeb34'
            
    },
        content_style={
        'borderLeft': '1px solid #d6d6d6',
        'borderRight': '1px solid #d6d6d6',
        'borderBottom': '1px solid #d6d6d6',
        'padding': '44px'
    },
        parent_style={
        'maxWidth': '1000px',
        'margin': '0 auto',
        'background': '#ffffff',
        'color':'#000000',
        'font-size': 'large'
    }
    ),
])


# In[715]:


fig11=go.Figure([go.Bar(x=list(df_ratings['sub'].values),y=list(df_ratings['rating'].values))])
fig12=go.Figure([go.Bar(x=list(df_rec['sub'].values),y=list(df_rec['semester'].values))])
fig13=go.Figure([go.Bar(x=["Advanced Web Technologies","Distributed Systems","Social Networks"],y=[1, 3, 4])])


# In[716]:


fig_sub=make_subplots(rows=1, cols=2)


# In[717]:


#fig_sub.add_trace(go.Bar(x=list(df_ratings['sub'].values),y=list(df_ratings['rating'].values)),row=1,col=1)


# In[718]:


@app.callback(Output('barchart', 'children'),
              [Input('dropdown_courses', 'value')])
def update_figure(value):
    if value is None:
        return html.Label('No value has been selected.Please select a value')
    elif value=='Rating':
        return html.Div(
                dcc.Graph(
                    id='bar chart',
                    figure=fig11,
                ),
                 
        )
    elif value=='Grade':
        return html.Div(
                dcc.Graph(
                    id='bar chart',
                    figure=fig12,
                )
        )
    elif value=='Semester':
        return html.Div(
                dcc.Graph(
                    id='bar chart',
                    figure=fig13,
                )
        )


# In[719]:



@app.callback(Output("content1", "figure"), [Input("button1", "n_clicks")],[State("checklist1", "value")])
def update_output(rows,value):
    if rows is not None:
       # Here, I could get DataFrames
        l2=recommend_courses(value)
        df1 = pd.DataFrame({'Courses' : value,
                                'Recommended Courses' : l2 }, 
                                columns=['Courses','Recommended Courses'])
        print(df1.columns)
    
      # But, have no ideas how to return dash_table.Databales instead ob go.Table
        
        table = ff.create_table(df1,height_constant=30)
        #table.layout.width=250
        for i in range(len(table.layout.annotations)):
            table.layout.annotations[i].font.size = 12
        return table
        


# In[720]:


if __name__ == '__main__':
    app.run_server(debug=False)


# In[597]:


app.layout=html.Div(style={'background-image': colors['background-image'],'margin-bottom':'10%'},children=[
    html.H1('Student Course Recommendation System', style={
            'textAlign': 'center', 'fontFamily': 'georgia','color':'#3269a8','background-color': 'lightblue'}),
    html.Div(id="tabs",style={'background-color':'white','height':'100%','width':'40%','margin-bottom':'20%','margin-left':'30%'}, children=[html.H2('Welcome to the Course Advisory System',
    style={'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'georgia','color':'#3269a8'}),html.H4('Select the List of Courses you would like to attend:',
    style={'textAlign': 'left','margin-left':'8%','margin': '48px 0', 'fontFamily': 'georgia','color':'#3269b8'}),dcc.Checklist(style={'font-size':'large','margin-left':'10%'},id='checklist1',
    options=[
         {'label': 'Advanced Image Synthesis', 'value': 'Advanced Image Synthesis'},
         {'label': 'Advanced Web Technologies', 'value': 'Advanced Web Technologies'},
         {'label': 'Cloud Web & Mobile', 'value': 'Cloud Web & Mobile'},
         {'label': 'Cognitive Robot Systems', 'value': 'Cognitive Robot Systems'},
         {'label': 'Computer Robot Vision', 'value': 'Computer Robot Vision'},
         {'label': 'Computer Graphics', 'value': 'Computer Graphics'},
         {'label': 'Gestaltung interaktiver Lehr-/Lern-Systeme', 'value': 'Gestaltung interaktiver Lehr-/Lern-Systeme'},
         {'label': 'Development of Safe and Secure Software', 'value': 'Development of Safe and Secure Software'},
         {'label': 'Digital Games Research', 'value': 'Digital Games Research'},
         {'label': 'Distributed Systems', 'value': 'Distributed Systems'},
         {'label': 'Electronic Communities and Social Networks', 'value': 'Electronic Communities and Social Networks'},
         {'label': 'Fault Diagnosis and Fault Tolerance in Technical Systems', 'value': 'Fault Diagnosis and Fault Tolerance in Technical Systems'},
         {'label': 'Formal Specification of Software Systems', 'value': 'Formal Specification of Software Systems'},
         {'label': 'Game Architecture and Design', 'value': 'Game Architecture and Design'},
         {'label': 'Information Engineering', 'value': 'Information Engineering'},
         {'label': 'Information Mining', 'value': 'Information Mining'},
         {'label': 'Information Retrieval', 'value': 'Information Retrieval'},
         {'label': 'Interaktive Systeme', 'value': 'Interaktive Systeme'},
         {'label': 'Internet of Things- Protocols and System Software', 'value': 'Internet of Things- Protocols and System Software'},
         {'label': 'Knowledge-Based Systems', 'value': 'Knowledge-Based Systems'},
         {'label': 'Learning Analytics', 'value': 'Learning Analytics'},
         {'label': 'Master Seminar Informatics', 'value': 'Master Seminar Informatics'},
         {'label': 'Modelling of Concurrent Systems', 'value': 'Modelling of Concurrent Systems'},
         {'label': 'Modelling-Analysis-Verification', 'value': 'Modelling-Analysis-Verification'},
         {'label': 'Natural-Language-based Human-Computer Interaction', 'value': 'Natural-Language-based Human-Computer Interaction'},
         {'label': 'Neurocomputing and Organic Computing', 'value': 'Neurocomputing and Organic Computing'},
         {'label': 'Pattern and Component based Software Development', 'value': 'Pattern and Component based Software Development'},
         {'label': 'Peer-to-Peer Systems', 'value': 'Peer-to-Peer Systems'},
         {'label': 'Recommender Systems', 'value': 'Recommender Systems'},
         {'label': 'Scientific Visualization', 'value': 'Scientific Visualization'},
         {'label': 'Test and Reliability of Digital Systems', 'value': 'Test and Reliability of Digital Systems'},
        
     ]),html.Button('Suggest ME!', id='button1',style={'color':'#4287f5','size':'large','margin-left':'10%','size':'150%'})])])
#     ,children=[
#         html.Div(style={'background-color': 'white','height':'100px','width':'40%','align':'center'}, children=[
#             html.H2('Welcome to the Course Advisory System',style={'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'georgia','color':'#3269a8'}),
#             html.H4('Select the List of Courses you would like to attend:',style={'textAlign': 'left', 'margin': '48px 0', 'fontFamily': 'georgia','color':'#3269b8'}),
#             #html.Div(dcc.Input(id='input1',value='Enter keyword',type='hidden')),
#             html.Div([dcc.Checklist(id='checklist1',
#     options=[
#         {'label': 'Advanced Image Synthesis', 'value': 'Advanced Image Synthesis'},
#         {'label': 'Advanced Web Technologies', 'value': 'Advanced Web Technologies'},
#         {'label': 'Cloud Web & Mobile', 'value': 'Cloud Web & Mobile'},
#         {'label': 'Cognitive Robot Systems', 'value': 'Cognitive Robot Systems'},
#         {'label': 'Computer Robot Vision', 'value': 'Computer Robot Vision'},
#         {'label': 'Computer Graphics', 'value': 'Computer Graphics'},
#         {'label': 'Gestaltung interaktiver Lehr-/Lern-Systeme', 'value': 'Gestaltung interaktiver Lehr-/Lern-Systeme'},
#         {'label': 'Development of Safe and Secure Software', 'value': 'Development of Safe and Secure Software'},
#         {'label': 'Digital Games Research', 'value': 'Digital Games Research'},
#         {'label': 'Distributed Systems', 'value': 'Distributed Systems'},
#         {'label': 'Electronic Communities and Social Networks', 'value': 'Electronic Communities and Social Networks'},
#         {'label': 'Fault Diagnosis and Fault Tolerance in Technical Systems', 'value': 'Fault Diagnosis and Fault Tolerance in Technical Systems'},
#         {'label': 'Formal Specification of Software Systems', 'value': 'Formal Specification of Software Systems'},
#         {'label': 'Game Architecture and Design', 'value': 'Game Architecture and Design'},
#         {'label': 'Information Engineering', 'value': 'Information Engineering'},
#         {'label': 'Information Mining', 'value': 'Information Mining'},
#         {'label': 'Information Retrieval', 'value': 'Information Retrieval'},
#         {'label': 'Interaktive Systeme', 'value': 'Interaktive Systeme'},
#         {'label': 'Internet of Things- Protocols and System Software', 'value': 'Internet of Things- Protocols and System Software'},
#         {'label': 'Knowledge-Based Systems', 'value': 'Knowledge-Based Systems'},
#         {'label': 'Learning Analytics', 'value': 'Learning Analytics'},
#         {'label': 'Master Seminar Informatics', 'value': 'Master Seminar Informatics'},
#         {'label': 'Modelling of Concurrent Systems', 'value': 'Modelling of Concurrent Systems'},
#         {'label': 'Modelling-Analysis-Verification', 'value': 'Modelling-Analysis-Verification'},
#         {'label': 'Natural-Language-based Human-Computer Interaction', 'value': 'Natural-Language-based Human-Computer Interaction'},
#         {'label': 'Neurocomputing and Organic Computing', 'value': 'Neurocomputing and Organic Computing'},
#         {'label': 'Pattern and Component based Software Development', 'value': 'Pattern and Component based Software Development'},
#         {'label': 'Peer-to-Peer Systems', 'value': 'Peer-to-Peer Systems'},
#         {'label': 'Recommender Systems', 'value': 'Recommender Systems'},
#         {'label': 'Scientific Visualization', 'value': 'Scientific Visualization'},
#         {'label': 'Test and Reliability of Digital Systems', 'value': 'Test and Reliability of Digital Systems'},
        
#     ],
#                                     #value=[]
#   )
#                      ],),
#             html.Button('Suggest ME!', id='button1',style={'color':'#4287f5','size':'large'}),
            
#           #dt.DataTable(id='Table',columns=['Course','Suggested Courses'])
#         ]),
        
#         ]),
       
#     ],
        
       
#     )


# In[ ]:





# In[ ]:




