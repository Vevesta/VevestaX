from email.utils import localtime
import pandas
import inspect
import os.path
import ipynbname
import random
import sys
import requests
import json
import matplotlib.pyplot as plt
import openpyxl
from pathlib import Path
import os, shutil

def test():
    return 'Test Executed Successfully'


class Experiment(object):
    def __init__(self):
        self._dataSourcing = None
        self._featureEngineering = None
        self._data=None
    
        self.startlocals = None
        self.variables = {}

    def get_filename(self):
        try:
            try:
                filename = ipynbname.name() + '.ipynb'

            except:
                try:
                    filePath = dict(inspect.getmembers(inspect.stack()[2][0]))['f_locals']['__file__']
                except:
                    filePath = sys.argv[0]
                filename = os.path.basename(filePath)
        except:
            filename = None

        return filename

    @property
    def dataSourcing(self):
        return self._dataSourcing

    @dataSourcing.setter
    def dataSourcing(self, value):
        if type(value) == pandas.core.frame.DataFrame:
            self._dataSourcing = value.columns
            self._data=value
            self._sampleSize=len(value)
            self._sampleSize=100 if self._sampleSize>=100 else self._sampleSize
            self._data=self._data.sample(self._sampleSize)
 
    @property
    def ds(self):
        return self._dataSourcing

    @dataSourcing.setter
    def ds(self, value):
        self.dataSourcing = value

    @property
    def featureEngineering(self):
        return self._featureEngineering

    @featureEngineering.setter
    def featureEngineering(self, value):
        if self._dataSourcing is None:
            print("Data Sourcing step missed.")
            if type(value) == pandas.core.frame.DataFrame:
                cols = value.columns
                self._featureEngineering = cols

        else:
            if type(value) == pandas.core.frame.DataFrame:
                cols = value.columns
                cols = cols.drop(self.dataSourcing)
                self._featureEngineering = cols

    @property
    def fe(self):
        return self._featureEngineering

    @featureEngineering.setter
    def fe(self, value):
        self.featureEngineering = value

    def startModelling(self):
        self.startlocals = dict(inspect.getmembers(inspect.stack()[1][0]))['f_locals'].copy()

    def endModelling(self):

        temp = dict(inspect.getmembers(inspect.stack()[1][0]))['f_locals'].copy()
        self.temp = inspect.getmembers(inspect.stack()[1])

        self.variables = {**self.variables, **{i: temp.get(i) for i in temp if
                                               i not in self.startlocals and i[0] != '_' and type(temp[i]) in [int,
                                                                                                               float,
                                                                                                               bool,
                                                                                                               str]}}

        return self.variables

    # create alias of method modellingStart and modellingEnd
    start = startModelling
    end = endModelling

    # Exp = Experiment
    # -------------
    def __getMessage(self):
        messagesList = ["For additional features, explore our tool at https://www.vevesta.com?utm_source=vevestaX for free.",
                        "Track evolution of Data Science projects at https://www.vevesta.com?utm_source=vevestaX for free.",
                        "Manage notes, codes and models in one single place by using our tool at https://www.vevesta.com?utm_source=vevestaX",
                        "For faster discovery of features, explore our tool at https://www.vevesta.com?utm_source=vevestaX",
                        "Find the right technique for your Machine Learning project at https://www.vevesta.com?utm_source=vevestaX"
                        ]
        return (messagesList[random.randint(0, len(messagesList) - 1)])

    def dump(self, techniqueUsed, filename=None, message=None, version=None, showMessage=True):

        existingData = None
        modelingData = None
        featureEngineeringData = None
        messageData = None
        experimentID = 1
        localTimestamp = localtime().strftime("%Y-%m-%d %H:%M:%S")
        mode = 'w'

        if (filename == None):
            filename = "vevesta.xlsx"
        print("Dumped the experiment in the file " + filename)

        # check if file already exists
        if (os.path.isfile(filename)):
            # mode = 'a'
            existingData = pandas.read_excel(filename, sheet_name='dataSourcing', index_col=[])
            featureEngineeringData = pandas.read_excel(filename, sheet_name='featureEngineering', index_col=[])
            modelingData = pandas.read_excel(filename, sheet_name='modelling', index_col=[])
            messageData = pandas.read_excel(filename, sheet_name='messages', index_col=[])
            experimentID = max(modelingData["experimentID"]) + 1

        if self._dataSourcing is None:
            df_dataSourcing = pandas.DataFrame(index=[1])
        else:
            df_dataSourcing = pandas.DataFrame(1, index=[1], columns=self._dataSourcing)

        df_dataSourcing.insert(0, 'experimentID', experimentID)
        df_dataSourcing = pandas.concat([existingData, df_dataSourcing], ignore_index=True).fillna(0)

        if self.featureEngineering is None:
            df_featureEngineering = pandas.DataFrame(index=[1])
        else:
            df_featureEngineering = pandas.DataFrame(1, index=[1], columns=self.featureEngineering)

        df_featureEngineering.insert(0, 'experimentID', experimentID)
        df_featureEngineering = pandas.concat([featureEngineeringData, df_featureEngineering],
                                              ignore_index=True).fillna(0)

        if self._dataSourcing is None and self._featureEngineering is None:
            modeling = pandas.DataFrame(
                data={**{'experimentID': experimentID, 'timestamp': localTimestamp},
                      **{k: [v] for k, v in self.variables.items()}}, index=[0])
        elif self._dataSourcing is None:
            modeling = pandas.DataFrame(data={
                **{'experimentID': experimentID, 'features': ','.join(self.featureEngineering),
                   'timestamp': localTimestamp}, **{k: [v] for k, v in self.variables.items()}},
                index=[0])
        elif self._featureEngineering is None:
            modeling = pandas.DataFrame(data={**{'experimentID': experimentID, 'features': ','.join(self.dataSourcing),
                                                 'timestamp': localTimestamp},
                                              **{k: [v] for k, v in self.variables.items()}}, index=[0])
        else:
            modeling = pandas.DataFrame(data={**{'experimentID': experimentID,
                                                 'features': ','.join(self.dataSourcing) + ',' + ','.join(
                                                     self.featureEngineering),
                                                 'timestamp': localTimestamp},
                                              **{k: [v] for k, v in self.variables.items()}}, index=[0])

        modeling = pandas.concat([modelingData, modeling], ignore_index=True)

        # message table
        data = {
            'experimentID': experimentID,
            'techniqueUsed': techniqueUsed,
            'message': message,
            'version': version,
            'filename': self.get_filename(),
            'timestamp': localTimestamp
        }

        df_messages = pandas.DataFrame(index=[1], data=data)
        df_messages = pandas.concat([messageData, df_messages], ignore_index=True)

        with pandas.ExcelWriter(filename, engine='openpyxl') as writer:

            df_dataSourcing.to_excel(writer, sheet_name='dataSourcing', index=False)

            df_featureEngineering.to_excel(writer, sheet_name='featureEngineering', index=False)
            modeling.to_excel(writer, sheet_name='modelling', index=False)

            df_messages.to_excel(writer, sheet_name='messages', index=False)
            pandas.DataFrame(self._data).to_excel(writer,sheet_name='sampledata',index=False)  

        # self.__plot(filename)

        if showMessage:
            message = self.__getMessage()
            print(message)


    def __plot(self, filename):

        modelingData = None

        if (filename == None):
            return print("Error: Provide the Excel File to plot the models")

        if (os.path.isfile(filename)):
            excelFile = openpyxl.load_workbook(filename, read_only=True)
            # check if modelling sheet exist in excel file
            if 'modelling' in excelFile.sheetnames:
                modelingData = pandas.read_excel(filename, sheet_name='modelling', index_col=[])
            else:
                return 
        
        # excluding numeric, datetime type
        nonNumericColumns = modelingData.select_dtypes(exclude=['number', 'datetime'])
        modelingData.drop(nonNumericColumns, axis=1, inplace=True)
        modelingData.drop('experimentID', axis=1, inplace=True)

        # checks if there are any columns after the timestamp column
        if len(modelingData.columns) == 0:
            return print("No variables to plot against Date")
        
        directoryToDumpData = 'vevestaXDump'
        self.__emptfldr(directoryToDumpData)
        # creating a new folder in current directory
        Path(directoryToDumpData).mkdir(parents=True, exist_ok=True)

        workbook = openpyxl.load_workbook(filename)
        workbook.create_sheet('performancePlots')
        plotSheet=workbook['performancePlots']
        xAxis = list(nonNumericColumns['timestamp'])
        columnValue = 2

        for column in modelingData.columns:
            yAxis = list(modelingData[column])

            imagename = str(column)+'.png'
            columntext = 'A'
            columntext += str(columnValue)

            # creates a seperate plots for every timestamp vs column and saves it
            fig,ax=plt.subplots()
            ax.plot(xAxis,yAxis)
            # rotating the x axis labels
            plt.xticks(rotation = 45)
            plt.title('Date vs '+str(column))
            plt.xlabel('Date')
            plt.ylabel(str(column))

            plt.savefig(os.path.join(directoryToDumpData,imagename), bbox_inches='tight') 

            img = openpyxl.drawing.image.Image(os.path.join(directoryToDumpData,imagename))  
            img.anchor = columntext
            plotSheet.add_image(img)

            columnValue+=20

        workbook.save(filename)

    # to turncate teh content inside the vevestaXDump folder if it exist
    def __emptfldr(self, dir_name):
        folder = dir_name
        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    def commit(self, techniqueUsed, filename=None, message=None, version=None, project_id=None):
        self.dump(techniqueUsed, filename=filename, message=message, version=version, showMessage=False)

        # api-endpoint
        token = open("access_token.txt", "r").read()
        backend_url = 'https://api.matrixkanban.com/services-1.0-SNAPSHOT/VevestaX'
        headers = {
            'Authorization': 'Bearer ' + token,
            'Access-Control-Allow-Origin': '*',
            'Accept': '*/*',
            'Content-Type': 'application/json'
        }
        payload = {
            "projectId": project_id,
            "title": techniqueUsed,
            "message": message,
            "modeling": self.variables,
            "dataSourced": self._dataSourcing.tolist(),
            "featureEngineered": self._featureEngineering.tolist()
        }
        response = requests.post(url=backend_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            print("Wrote experiment to tool, vevesta")
        else:
            print("Failed to write experiment to tool, vevesta")


