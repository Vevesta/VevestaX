import datetime
import pandas
import inspect
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
        self.__dataSourcing = None
        self.__featureEngineering = None
        self.__data=None
        self.__correlation = None

        self.__primitiveDataTypes = [int, str, float, bool]
        self.__startlocals = None
        self.__variables = {}
        self.__filename = self.get_filename()
        self.__sampleSize = 0

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
        return self.__dataSourcing

    @dataSourcing.setter
    def dataSourcing(self, value):
        if type(value) == pandas.core.frame.DataFrame:
            self.__dataSourcing = value.columns
            self.__data=value
            self.__sampleSize=len(value)
            self.__correlation = value.corr(method='pearson')

    @property
    def ds(self):
        return self.__dataSourcing

    @dataSourcing.setter
    def ds(self, value):
        self.dataSourcing = value

    @property
    def featureEngineering(self):
        return self.__featureEngineering

    @featureEngineering.setter
    def featureEngineering(self, value):
        if self.__dataSourcing is None:
            print("Data Sourcing step missed.")
            if type(value) == pandas.core.frame.DataFrame:
                cols = value.columns
                self.__featureEngineering = cols

        else:
            if type(value) == pandas.core.frame.DataFrame:
                cols = value.columns
                cols = [col for col in cols if col not in self.__dataSourcing]
                #cols = cols.drop(self.dataSourcing)
                self.__featureEngineering = cols

        if type(value) == pandas.core.frame.DataFrame:
            self.__correlation = value.corr(method='pearson')

    @property
    def fe(self):
        return self.__featureEngineering

    @featureEngineering.setter
    def fe(self, value):
        self.featureEngineering = value

    def startModelling(self):
        self.__startlocals = dict(inspect.getmembers(inspect.stack()[1][0]))['f_locals'].copy()

    def endModelling(self):

        temp = dict(inspect.getmembers(inspect.stack()[1][0]))['f_locals'].copy()
        self.temp = inspect.getmembers(inspect.stack()[1])

        self.__variables = {**self.__variables, **{i: temp.get(i) for i in temp if
                                               i not in self.__startlocals and i[0] != '_' and (type(temp[i]) in self.__primitiveDataTypes or isinstance(temp[i], (str,int,float,bool)))}}

        return self.__variables

    # create alias of method modellingStart and modellingEnd
    start = startModelling
    end = endModelling

    # function to get arguments of a function
    def param(self, **decoratorparam):
        def params(functionName):
            def wrapper(*args, **kwargs):
                # to get parameters of function that are passed
                functionParameters = inspect.signature(functionName).bind(*args, **kwargs).arguments
                functionParameters = dict(functionParameters)
                # to get values that are not passed and defautlt values
                defaultParameters = inspect.signature(functionName)
                for param in defaultParameters.parameters.values():
                    # checks in key exist in abouve dictionary then doesn't update it will default value, otherwise append into dictionary
                    if ( param.default is not param.empty) and (param.name not in functionParameters):
                        functionParameters[param.name] = param.default

                for key, value in decoratorparam.items():
                    if (key in functionParameters) and (type[value] in self.__primitiveDataTypes):
                        functionParameters[value] = functionParameters.pop(key)

                self.__variables = {**self.__variables, **{key: value for key, value in functionParameters.items() if type(value) in [int, float, bool, str] and key not in self.__variables}}

            return wrapper
        return params

    # Exp = Experiment
    # -------------
    def __getMessage(self):
        messagesList = ["For additional features, explore our tool at https://www.vevesta.com?utm_source=vevestaX for free.",
                        "Track evolution of Data Science projects at https://www.vevesta.com?utm_source=vevestaX for free.",
                        "Manage notes, codes and models in one single place by using our tool at https://www.vevesta.com?utm_source=vevestaX",
                        "For faster discovery of features, explore our tool at https://www.vevesta.com?utm_source=vevestaX",
                        "Find the right technique for your Machine Learning project at https://www.vevesta.com?utm_source=vevestaX",
                        "Give us a reason to celebrate, give us your feedback at vevestax@vevesta.com",
                        "Give us a reason to cheer, give us a star on Github: https://github.com/Vevesta/VevestaX",
                        "Mail us at vevestax@vevesta.com to follow the latest updates to the library, VevestaX.",
                        "Help your ML community work better by giving us a Github star at https://github.com/Vevesta/VevestaX",
                        "Easily organize and manage your notes, documents, code, data and models. Explore Vevesta at https://www.vevesta.com?utm_source=vevestaX",
                        "Spread the word in your ML community by giving us a Github star at https://github.com/Vevesta/VevestaX",
                        "Love VevestaX? Give us a shoutout at vevestax@vevesta.com",
                        "Get access to latest release ahead of others by subscribing to vevestax@vevesta.com"
                        ]
        return (messagesList[random.randint(0, len(messagesList) - 1)])

    def __colorCellExcel(self, val):
        if -1 <= val <= -0.9:
            color = '#0D47A1'
            return 'background-color: %s' % color
        elif -0.9 < val <= -0.8:
            color = '#1565C0'
            return 'background-color: %s' % color
        elif -0.8 < val <= -0.7:
            color = '#1976D2'
            return 'background-color: %s' % color
        elif -0.7 < val <= -0.6:
            color = '#2962FF'
            return 'background-color: %s' % color
        elif -0.6 < val <= -0.5:
            color = '#2979FF'
            return 'background-color: %s' % color
        elif -0.5 < val <= -0.4:
            color = '#1E88E5'
            return 'background-color: %s' % color
        elif -0.4 < val <= -0.3:
            color = '#42A5F5'
            return 'background-color: %s' % color
        elif -0.3 < val <= -0.2:
            color = '#64B5F6'
            return 'background-color: %s' % color
        elif -0.2 < val <= -0.1:
            color = '#90CAF9'
            return 'background-color: %s' % color
        elif -0.1 < val <= 0:
            color = '#BBDEFB'
            return 'background-color: %s' % color
        elif val == 0:
            color = '#E3F2FD'
            return 'background-color: %s' % color
        elif 0 < val <= 0.1:
            color = '#F1F8E9'
            return 'background-color: %s' % color
        elif 0.1 < val <= 0.2:
            color = '#DCEDC8'
            return 'background-color: %s' % color
        elif 0.2 < val <= 0.3:
            color = '#C5E1A5'
            return 'background-color: %s' % color
        elif 0.3 < val <= 0.4:
            color = '#AED581'
            return 'background-color: %s' % color
        elif 0.4 < val <= 0.5:
            color = '#9CCC65'
            return 'background-color: %s' % color
        elif 0.5 < val <= 0.6:
            color = '#7CB342'
            return 'background-color: %s' % color
        elif 0.6 < val <= 0.7:
            color = '#689F38'
            return 'background-color: %s' % color
        elif 0.7 < val <= 0.8:
            color = '#558B2F'
            return 'background-color: %s' % color
        elif 0.8 < val <= 0.9:
            color = '#33691E'
            return 'background-color: %s' % color
        elif 0.9 < val <= 1:
            color = '#2E7D32'
            return 'background-color: %s' % color

    def __textColor(self, val):
        if -0.1 < val < 0.1:
            color = 'black'
        else:
            color = 'white'
        return 'color: %s' % color

    def dump(self, techniqueUsed, filename=None, message=None, version=None, showMessage=True):

        existingData = None
        modelingData = None
        featureEngineeringData = None
        messageData = None
        experimentID = 1
        localTimestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = 'w'

        if (filename == None):
            filename = "vevesta.xlsx"


        #updating variables
        #when no V.start & v.end are not called, all variables in the code get tracked or in colab/kaggle where all variables will get tracked
        if(len(self.__variables) == 0):
            temp = dict(inspect.getmembers(inspect.stack()[1][0]))['f_locals'].copy()
            self.__variables = { **{i: temp.get(i) for i in temp if i[0] != '_' and (type(temp[i]) in self.__primitiveDataTypes or isinstance(temp[i], (str,int,float,bool)))}}

        # check if file already exists
        if (os.path.isfile(filename)):
            # mode = 'a'
            existingData = pandas.read_excel(filename, sheet_name='dataSourcing', index_col=[])
            featureEngineeringData = pandas.read_excel(filename, sheet_name='featureEngineering', index_col=[])
            modelingData = pandas.read_excel(filename, sheet_name='modelling', index_col=[])
            messageData = pandas.read_excel(filename, sheet_name='messages', index_col=[])
            experimentID = max(modelingData["experimentID"]) + 1

        if self.__dataSourcing is None:
            df_dataSourcing = pandas.DataFrame(index=[1])
        else:
            df_dataSourcing = pandas.DataFrame(1, index=[1], columns=self.__dataSourcing)

        df_dataSourcing.insert(0, 'experimentID', experimentID)
        df_dataSourcing = pandas.concat([existingData, df_dataSourcing], ignore_index=True).fillna(0)

        if self.featureEngineering is None:
            df_featureEngineering = pandas.DataFrame(index=[1])
        else:
            df_featureEngineering = pandas.DataFrame(1, index=[1], columns=self.featureEngineering)

        df_featureEngineering.insert(0, 'experimentID', experimentID)
        df_featureEngineering = pandas.concat([featureEngineeringData, df_featureEngineering],
                                              ignore_index=True).fillna(0)

        if self.__dataSourcing is None and self.__featureEngineering is None:
            modeling = pandas.DataFrame(
                data={**{'experimentID': experimentID, 'timestamp': localTimestamp},
                      **{k: [v] for k, v in self.__variables.items()}}, index=[0])
        elif self.__dataSourcing is None:
            modeling = pandas.DataFrame(data={
                **{'experimentID': experimentID, 'features': ','.join(self.featureEngineering),
                   'timestamp': localTimestamp}, **{k: [v] for k, v in self.__variables.items()}},
                index=[0])
        elif self.__featureEngineering is None:
            modeling = pandas.DataFrame(data={**{'experimentID': experimentID, 'features': ','.join(self.dataSourcing),
                                                 'timestamp': localTimestamp},
                                              **{k: [v] for k, v in self.__variables.items()}}, index=[0])
        else:
            modeling = pandas.DataFrame(data={**{'experimentID': experimentID,
                                                 'features': ','.join(self.dataSourcing) + ',' + ','.join(
                                                     self.featureEngineering),
                                                 'timestamp': localTimestamp},
                                              **{k: [v] for k, v in self.__variables.items()}}, index=[0])

        modeling = pandas.concat([modelingData, modeling], ignore_index=True)

        # message table
        data = {
            'experimentID': experimentID,
            'techniqueUsed': techniqueUsed,
            'message': message,
            'version': version,
            'filename': self.__filename,
            'timestamp': localTimestamp
        }

        df_messages = pandas.DataFrame(index=[1], data=data)
        df_messages = pandas.concat([messageData, df_messages], ignore_index=True)

        self.__sampleSize=100 if self.__sampleSize>=100 else self.__sampleSize
        sampledData=self.__data.sample(self.__sampleSize)

        with pandas.ExcelWriter(filename, engine='openpyxl') as writer:

            df_dataSourcing.to_excel(writer, sheet_name='dataSourcing', index=False)

            df_featureEngineering.to_excel(writer, sheet_name='featureEngineering', index=False)
            modeling.to_excel(writer, sheet_name='modelling', index=False)

            df_messages.to_excel(writer, sheet_name='messages', index=False)
            pandas.DataFrame(sampledData).to_excel(writer,sheet_name='sampledata',index=False)

            if self.__correlation is not None:
                pandas.DataFrame(self.__correlation).style.\
                applymap(self.__colorCellExcel).\
                applymap(self.__textColor).\
                to_excel(writer, sheet_name='EDA-correlation', index=True)

        self.__missingEDAValues(filename)

        self.__plot(filename)

        print("Dumped the experiment in the file " + filename)

        if showMessage:
            message = self.__getMessage()
            print(message)

    def __missingEDAValues(self, fileName):

        if self.__data.empty or len(self.__data)==0:
            return

        if (fileName == None):
            return print("Error: Provide the Excel File to plot the models")

        columnTextImgone = 'B2'
        columnTextImgtwo = 'B44'
        directoryToDumpData = 'vevestaXDump'
        # creating a new folder in current directory
        Path(directoryToDumpData).mkdir(parents=True, exist_ok=True)
        ValueImageFile = "Value.png"
        ValueRatioImageFile = "ValuePerFeature.png"
        NumericalFeatureDistributionImageFile = "NumericalFeatureDistribution.png"
        NonNumericFeaturesImgFile = "NonNumericFeatures.png"

        plt.figure(figsize=(13,8))
        plt.imshow(self.__data.isna(), aspect="auto", interpolation="nearest", cmap="coolwarm", extent=[0,7,0,7])
        plt.title("Sample Number vs Column Number")
        plt.xlabel("Column Number")
        plt.ylabel("Sample Number")
        plt.savefig(os.path.join(directoryToDumpData,ValueImageFile),bbox_inches='tight', dpi=100)
        plt.close()

        RatioData = self.__data.isna().mean().sort_values()
        xAxis = list(RatioData.index)
        yAxis = list(RatioData)
        plt.figure(figsize=(13,11))
        plt.bar(xAxis,yAxis)
        plt.title("Percentage of missing values per feature")
        plt.xlabel("Feature Names")
        plt.ylabel("Ratio of missing values per feature")
        plt.savefig(os.path.join(directoryToDumpData,ValueRatioImageFile),bbox_inches='tight', dpi=100)
        plt.close()


        self.__data.plot(lw=0,marker="x",subplots=True,layout=(-1, 4),figsize=(20, 25),markersize=5, title="Numeric feature Distribution").flatten()
        plt.savefig(os.path.join(directoryToDumpData,NumericalFeatureDistributionImageFile),bbox_inches='tight', dpi=100)
        plt.close()

        # Identify non-numerical features
        nonNumericalColumns = self.__data.select_dtypes(exclude=["number", "datetime"])
        if len(nonNumericalColumns.columns) is not 0:
            # Create figure object with 3 subplots
            fig, axes = plt.subplots(ncols=1, nrows=len(nonNumericalColumns.columns), figsize=(18, 20))
            # Loop through features and put each subplot on a matplotlib axis object
            for col, ax in zip(nonNumericalColumns.columns, axes.ravel()):
                # Selects one single feature and counts number of unique value and Plots this information in a figure with log-scaled y-axis
                nonNumericalColumns[col].value_counts().plot(logy=True, title=col, lw=0, marker="X", ax=ax, markersize=5)
                # plt.tight_layout()
                plt.savefig(os.path.join(directoryToDumpData,NonNumericFeaturesImgFile),bbox_inches='tight', dpi=100)
            plt.close()

        if (os.path.isfile(fileName)):
            workBook = openpyxl.load_workbook(fileName)
            workBook.create_sheet('EDA-missingValues')
            plotSheet=workBook['EDA-missingValues']
            img = openpyxl.drawing.image.Image(os.path.join(directoryToDumpData,ValueImageFile))
            img.anchor = columnTextImgone
            plotSheet.add_image(img)
            image = openpyxl.drawing.image.Image(os.path.join(directoryToDumpData,ValueRatioImageFile))
            image.anchor = columnTextImgtwo
            plotSheet.add_image(image)

            # adding the plot for the Numeric Fetaure Distribution
            workBook.create_sheet('EDA-NumericfeatureDistribution')
            fetaureplotsheet = workBook['EDA-NumericfeatureDistribution']
            featureImg = openpyxl.drawing.image.Image(os.path.join(directoryToDumpData,NumericalFeatureDistributionImageFile))
            featureImg.anchor = columnTextImgone
            fetaureplotsheet.add_image(featureImg)

            # adding non-numeric column
            if os.path.exists(os.path.join(directoryToDumpData,NonNumericFeaturesImgFile)):
                workBook.create_sheet('EDA-NonNumericFeatures')
                nonNumericPlotSheet = workBook['EDA-NonNumericFeatures']
                nonNumericFeatureImage = openpyxl.drawing.image.Image(os.path.join(directoryToDumpData,NonNumericFeaturesImgFile))
                nonNumericFeatureImage.anchor = columnTextImgone
                nonNumericPlotSheet.add_image(nonNumericFeatureImage)

            workBook.save(fileName)
        workBook.close()

    def __plot(self, fileName):

        modelingData = None

        if (fileName == None):
            return print("Error: Provide the Excel File to plot the models")

        sheetName = 'modelling'
        modelingData = self.__getExcelSheetData(fileName, sheetName)

        # returns nothing if dataframe is empty
        if modelingData.empty:
            return

        # excluding numeric, datetime type
        nonNumericColumns = modelingData.select_dtypes(exclude=['number', 'datetime'])
        modelingData.drop(nonNumericColumns, axis=1, inplace=True)
        modelingData.drop('experimentID', axis=1, inplace=True)

        # checks if there are any columns after the timestamp column
        if len(modelingData.columns) == 0:
            return

        directoryToDumpData = 'vevestaXDump'
        self.__truncateFolder(directoryToDumpData)
        # creating a new folder in current directory
        Path(directoryToDumpData).mkdir(parents=True, exist_ok=True)

        # checks if file exist then only loads it and create a new sheet for plots
        if (os.path.isfile(fileName)):
            workBook = openpyxl.load_workbook(fileName)
            workBook.create_sheet('performancePlots')
            plotSheet=workBook['performancePlots']
            xAxis = list(nonNumericColumns['timestamp'])
            columnValue = 2

            for column in modelingData.columns:
                yAxis = list(modelingData[column])

                imageName = str(column)+'.png'
                columnText = 'B'
                columnText += str(columnValue)

                # creates a seperate plots for every timestamp vs column and saves it
                fig,ax=plt.subplots()
                ax.plot(xAxis,yAxis, linestyle='-', marker='o')
                # size of plot horizontally fixed to 5 inches and height to 5 inches
                plt.gcf().set_size_inches(13,5)
                # rotating the x axis labels
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')

                plt.title('Timestamp vs '+str(column))
                plt.xlabel('Timestamp')
                plt.ylabel(str(column))

                plt.savefig(os.path.join(directoryToDumpData,imageName), bbox_inches='tight', dpi=100)
                plt.close()

                img = openpyxl.drawing.image.Image(os.path.join(directoryToDumpData,imageName))
                img.anchor = columnText
                plotSheet.add_image(img)


                columnValue+=27

            workBook.save(fileName)
        workBook.close()

    # to truncate the content inside the vevestaXDump folder if it exist
    def __truncateFolder(self, folderName):
        if os.path.isdir(folderName):
            for filename in os.listdir(folderName):
                file_path = os.path.join(folderName, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    # generic function to get modelling sheet data from any excel file with sheetName modelling
    def __getExcelSheetData(self, fileName, sheetName):
        # checks if file exist then only if fetches modelling data
        if (os.path.isfile(fileName)):
            excelFile = openpyxl.load_workbook(fileName, read_only=True)
                # check if modelling sheet exist in excel file
            if sheetName in excelFile.sheetnames:
                modelingData = pandas.read_excel(fileName, sheet_name=sheetName, index_col=[])
                return modelingData

    def commit(self, techniqueUsed, filename=None, message=None, version=None, projectId=None, attachmentFlag=True):
        self.dump(techniqueUsed, filename=filename, message=message, version=version, showMessage=False)

        # api-endpoint
        token = open("access_token.txt", "r").read()
        backend_url = 'https://api.matrixkanban.com/services-1.0-SNAPSHOT'

        # upload attachment
        filename = self.get_filename()
        file_exists = os.path.exists(filename)
        if attachmentFlag:
            if file_exists:
                files = {'file': open(filename, 'rb')}
                headers_for_file = {'Authorization': 'Bearer '+token}
                params = {'taskId': 0}
                response = requests.post(url=backend_url+'/Attachments', headers=headers_for_file, params=params, files=files)
                attachments = list()
                attachments.append(response.json())

        # upload note
        headers_for_note = {
            'Authorization': 'Bearer ' + token,
            'Access-Control-Allow-Origin': '*',
            'Accept': '*/*',
            'Content-Type': 'application/json'
        }
        payload = {
            "projectId": projectId,
            "title": techniqueUsed,
            "message": message,
            "modeling": self.__variables,
            "dataSourced": self.__dataSourcing.tolist(),
            "featureEngineered": self.__featureEngineering.tolist()
        }
        if attachmentFlag:
            if file_exists:
                payload['attachments'] = attachments
            else:
                payload['errorMessage'] = 'File not pushed to Vevesta'
                print('File not pushed to Vevesta')

        response = requests.post(url=backend_url+'/VevestaX', headers=headers_for_note, data=json.dumps(payload))

        if response.status_code == 200:
            print("Wrote experiment to tool, Vevesta")
        else:
            print("Failed to write experiment to tool, Vevesta")
