import pandas as pd
import numpy as np
import pyodbc
from scipy.optimize import linprog
pd.set_option('display.max_columns', None)

class Scheduler():
    """

    :param powerMaxLimits: List of max power limits for units in following order:
        maxZTPOK,
        MaxOgCHP,
        MaxECII,
        MaxOgCoal,
        MaxBB,
        MaxECI,
    :param powerMinLimits: List of min power limits for units in following order:
        minZTPOK,
        MinOgCHP,
        MinECII,
        MinOgCoal,
        MinBB,
        MinECI,
    :param unitsCosts:List of operation costs for units in following order:
        ZTPOK,
        OgCHP,
        ECII,
        OgCoal
        BB,
        ECI
    :param unitsPriorities: Dictionary with units operation priorities.
    :param lowerAmbTempBoundaryZTPOK: lower ambient temperature boundary where ZTPOK works with fixed
    lowerBoundaryPowerZTPOK power,
    :param upperAmbTempBoundaryZTPOK: higher ambient temperature boundary where ZTPOK works with fixed
    upperBoundaryPowerZTPOK power,
    :param lowerBoundaryPowerZTPOK: ZTPOK fixed power generated below lowerAmbTempBoundaryZTPOK temperature,
    :param upperBoundaryPowerZTPOK: ZTPOK fixed power generated above upperAmbTempBoundaryZTPOK temperature,
    :param ambTempBoundaryECI: boundary ambient temperature below which ECI should be turned on at
    lowerBoundaryPowerECI power level
    :param lowerBoundaryPowerECI: min power for ECI unit below lowerBoundaryPowerECI ambient temperature,
    :param powerLimitFactorECII: factor which can be used to increase CurrentECIIPower = f(TempAmbient)
    :param eciiExtraPowerFraction: factor which can be used to use additional ECII power when ECI is working on full power to not run next units.
    Additional power will be a fraction of CurrentECIIPower.


    \n Assumptions:
        \n 1. Current max power for ECII is calculated based on functions defined using historical data so
        CurrentECIIPower = f(TempAmbient). You can increase or decrease the value of CurrentECIIPower using
        powerLimitFactorECII factor which by default is equal to 1.

        \n 2. ECI works at the level of min lowerBoundaryPowerECI (by default 8 MW) if ambient temperature is below
        value of attribute ambTempBoundaryECI (by default 12 Cdeg),

        \n 3. In unitsPriorities you are defining priority for ECII_II which means that ECII unit can use more power
        than it results from CurrentECIIPower.
        By default we assume that it is not working so eciiExtraPowerFraction is euql to 0, however you can use
        every value as it's fraction of CurrentECIIPower.

        \n 4. ZTPOK operates in three ambient temperature ranges defined by lowerAmbTempBoundaryZTPOK by
        default equal to 6 and upperAmbTempBoundaryZTPOK by default equal to 15. If ambient temperature is below
        lowerAmbTempBoundaryZTPOK ZTPOK operates with power equal to lowerBoundaryPowerZTPOK, when ambient
        temperaure is above upperAmbTempBoundaryZTPOK then operates with power equal to upperBoundaryPowerZTPOK
        but when it's between lowerAmbTempBoundaryZTPOK and upperAmbTempBoundaryZTPOK ZTPOK operates
        with power obtained from linear function defined by defineLineCoeff method.
    """

    def __init__(self, powerMaxLimits = [28.2, 1.5, 325, 13.1, 36.7, 38],
                 powerMinLimits=[0,0,0,0,0,0],
                 unitsCosts = [201, 202, 237, 469, 469, 512],
                 unitsPriorities = {'ZTPOK' : 1, 'OgCHP' : 2, 'OgCoal': 6, 'BB' : 7,'ECI' : 4, 'ECII_I' : 3, 'ECII_II' : 5 },
                 windowSize = 4,
                 lowerAmbTempBoundaryZTPOK = 6,
                 upperAmbTempBoundaryZTPOK = 15,
                 lowerBoundaryPowerZTPOK = 28.2,
                 upperBoundaryPowerZTPOK = 18,
                 ambTempBoundaryECI = 12,
                 lowerBoundaryPowerECI = 8,
                 powerLimitFactorECII = 1,
                 eciiExtraPowerFraction = 0.2):


        self.powerMaxZTPOK,  self.powerMaxOgCHP, self.powerMaxECII_II, self.powerMaxOgCoal, self.powerMaxBB,  self.powerMaxECI = powerMaxLimits
        self.powerMinZTPOK, self.powerMinOgCHP, self.powerMinECII_II, self.powerMinOgCoal, self.powerMinBB, self.powerMinECI = powerMinLimits
        self.costZTPOK, self.costOgCHP, self.costECII_I, self.costOgCoal, self.costBB, self.costECI = unitsCosts
        self.costECII_II = self.costECII_I  # we are assuming that their costs are the same
        self.unitsPriorities = list(unitsPriorities.values())
        self.lowerAmbTempBoundaryZTPOK = lowerAmbTempBoundaryZTPOK
        self.upperAmbTempBoundaryZTPOK = upperAmbTempBoundaryZTPOK
        self.lowerBoundaryPowerZTPOK = lowerBoundaryPowerZTPOK
        self.upperBoundaryPowerZTPOK = upperBoundaryPowerZTPOK
        self.ambTempBoundaryECI = ambTempBoundaryECI
        self.lowerBoundaryPowerECI = lowerBoundaryPowerECI
        self.powerLimitFactorECII = powerLimitFactorECII
        self.eciiExtraPowerFraction = eciiExtraPowerFraction
        self.windowSize = windowSize

    def defineLineCoeff(self,xCoordinates, yCoordinates):
        """
        Function which calculates slope coefficients of a line

        :param xCoordinates:  list/tuple of points x coordinates
        :param yCoordinates: list/tuple of points y coordinates
        :return:
            a,b - linear function factor from equation y = ax + b
        """

        xA, xB = xCoordinates
        yA, yB = yCoordinates

        assert xA != xB, f"Temperature limits/bounds {self.lowerAmbTempBoundaryZTPOK} and {self.upperAmbTempBoundaryZTPOK} for ZTPOK power curve are equal!"

        a = (yA - yB) / (xA - xB)
        b = yA - ((yA - yB) / (xA - xB)) * xA

        return a, b

    def getPowerLimits(self, dataframe):
        """
        Function which adds columns with min, max power for specified units.

        :param dataframe: Dataframe with temperature forecast defined in column PrognozaTemperatury, and load forecast for Miasto (PrognozaObciazeniaMiasto column) and Fordon area (PrognozaObciazeniaFordon column).
        :return: Input dataframe with additional columns with min, max units power for a specified units.
        """
        conditions = [dataframe['PrognozaTemperatury'] < self.lowerAmbTempBoundaryZTPOK,
                      dataframe['PrognozaTemperatury'] > self.upperAmbTempBoundaryZTPOK]

        choices = [self.lowerBoundaryPowerZTPOK,
                   self.upperBoundaryPowerZTPOK]

        a,b = self.defineLineCoeff(xCoordinates = [self.lowerAmbTempBoundaryZTPOK, self.upperAmbTempBoundaryZTPOK]
                                   ,yCoordinates = [self.lowerBoundaryPowerZTPOK, self.upperBoundaryPowerZTPOK])

        default = dataframe['PrognozaTemperatury'] * (a) + (b)

        return (dataframe
            .assign(
            MocMaxZTPOK =np.select(conditions, choices, default=default),
            MocMinZTPOK = self.powerMinZTPOK,
            MocMaxOgCHP = self.powerMaxOgCHP,
            MocMinOgCHP=self.powerMinOgCHP,
            MocMaxOgCoal = self.powerMaxOgCoal,
            MocMinOgCoal=self.powerMinOgCoal,
            MocMaxBB = self.powerMaxBB,
            MocMinBB=self.powerMinBB,
            MocMaxECI = self.powerMaxECI,
            MocMinECI = lambda df_: np.where(df_['PrognozaTemperatury'] < self.ambTempBoundaryECI,
                                           self.lowerBoundaryPowerECI,
                                             self.powerMinECI),
            MocMaxECII_I = lambda df_: (((df_['PrognozaObciazeniaMiasto'] / (
                        df_['PrognozaObciazeniaMiasto'] + df_['PrognozaObciazeniaFordon'])) *
                                       (4872.788 * (df_['PrognozaTemperatury'] + 273.15) ** 3 +
                                        (-4117244.609) * (df_['PrognozaTemperatury'] + 273.15) ** 2 +
                                        (1150800997.586) * (df_['PrognozaTemperatury'] + 273.15) ** 1
                                        - 106209288742.75)) / 1000000) * self.powerLimitFactorECII,
            MocMaxECII_II = lambda df_: df_['MocMaxECII_I'] * self.eciiExtraPowerFraction,
            MocMinECII_I = self.powerMinECII_II
                )
            )

    def getCosts(self,dataframe):
        """
        Function which adds columns with operational costs for specified units.

        :param dataframe: Dataframe with temperature forecast defined in column PrognozaTemperatury, and load forecast for Miasto (PrognozaObciazeniaMiasto column) and Fordon area (PrognozaObciazeniaFordon column).
        :return:  Input dataframe with additional columns with operational costs for a specified units.
        """

        return (dataframe
            .assign(
            costZTPOK = self.costZTPOK,
            costOgCHP = self.costOgCHP,
            costOgCoal = self.costOgCoal,
            costBB = self.costBB,
            costECI = self.costECI,
            costECII_I = self.costECII_I,
            costECII_II = self.costECII_II
                )
            )

    def prepareDataForOptimization(self, dataframe):
        """
        Function which transform input data from columns PrognozaTemperatury, PrognozaObciazeniaMiasto and
        PrognozaObciazeniaFordon into float64 type

        :param dataframe: input dataframe with PrognozaTemperatury, PrognozaObciazeniaMiasto and
        PrognozaObciazeniaFordon columns.
        :return: dataframe with modified columns.
        """

        df = dataframe.copy()

        return (df
                .assign(
            PrognozaTemperatury=lambda df_: df_['PrognozaTemperatury']
                                                .astype('string')
                                                .str
                                                .replace(",", ".")
                                                .astype('float64') - (0),
            PrognozaObciazeniaMiasto=lambda df_: df_['PrognozaObciazeniaMiasto']
                                                .astype('string')
                                                .str
                                                .replace(",", ".")
                                                .astype('float64') + (0),
            PrognozaObciazeniaFordon=lambda df_: df_['PrognozaObciazeniaFordon']
                                                .astype('string')
                                                .str
                                                .replace(",", ".")
                                                .astype('float64')
        )
                .pipe(self.getPowerLimits)
                .pipe(self.getCosts)
                )

    def optimizer(self, row):
        """
        Function which is responsible for optimization process of a operational cost for a given hour (row in dataframe)
        in input time series.


        :param row: row from a dataframe with load forecast, weather forecast, min/max power and operational costs
        of units

        :return: pandas series with powers of units and total operational cost for a given hour, using specified units.
        """

        demand = row['PrognozaObciazeniaMiasto']
        maxPowers = [
            row['MocMaxZTPOK'],
            row['MocMaxOgCHP'],
            row['MocMaxOgCoal'],
            row['MocMaxBB'],
            row['MocMaxECI'],
            row['MocMaxECII_I'],
            row['MocMaxECII_II']
            ]

        minPowers = [
            row['MocMinZTPOK'],
            row['MocMinOgCHP'],
            row['MocMinOgCoal'],
            row['MocMinBB'],
            row['MocMinECI'],
            row['MocMinECII_I'],
            row['MocMinECII_I']  #DUPLICATION ON PURPOSE
            ]
        costs = [
            row['costZTPOK'],
            row['costOgCHP'],
            row['costOgCoal'],
            row['costBB'],
            row['costECI'],
            row['costECII_I'],
            row['costECII_II']
            ]

        # Define the number of original decision variables (power generation) and binary decision variables (on/off status)
        numOriginalVars = len(costs)
        numBinaryVars = numOriginalVars

        # Objective function: minimize total cost weighted by priority
        c = [self.unitsPriorities[i] * costs[i] for i in range(len(costs))]
        c.extend([0] * numBinaryVars)  # Add zeros for the binary decision variables

        # Inequality constraints: unit power <= max power
        A_ub = np.eye(len(costs))

        # Reshape A_ub to match the expected dimensions
        A_ub = np.vstack([A_ub, np.zeros((A_ub.shape[1], A_ub.shape[0]))]).T
        b_ub = maxPowers  # Upper bounds for power generation

        # Equality constraint: total power generated = demand
        A_eq = np.array([[1] * len(costs) + [0] * numBinaryVars])
        b_eq = [demand]

        # Bounds for each unit's power generation and the binary decision variables (0 or 1)
        bounds = [(minPower, maxPower ) for minPower, maxPower in zip(minPowers, maxPowers)]

        bounds.extend([(0, 1) for _ in range(numBinaryVars)])  # Binary decision variables

        # Solve the mixed-integer linear programming problem
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            powerGenerated = result.x[:numOriginalVars]  # Extract the power generated
            totalCost = np.dot(powerGenerated, costs)  # Calculate total cost
            return pd.Series({
                'ZTPOK': powerGenerated[0],
                'OgCHP': powerGenerated[1],
                'OgCoal': powerGenerated[2],
                'BB': powerGenerated[3],
                'ECI': powerGenerated[4],
                'ECII_I': powerGenerated[5],
                'ECII_II': powerGenerated[6],
                'TotalCost': totalCost
            })
        else:
            return pd.Series({
                'ZTPOK': None,
                'OgCHP': None,
                'OgCoal': None,
                'BB': None,
                'ECI': None,
                'ECII_I': None,
                'ECII_II': None,
                'TotalCost': None})

    def runOptimizer(self, df):

        """
        Function which run the optimizer and create result dataframe

        :param df: input dataframe with load forecast, weather forecast, min/max power and operational costs
        of units
        :return: dataframe with results in specific structure
        """

        resultDf = df.apply(self.optimizer, axis=1)
        finalDf= pd.concat([df, resultDf], axis=1)

        dfReturn = (finalDf
            .assign(
            ECII=lambda df_: df_.ECII_I + df_.ECII_II,
            turnOff = lambda df_: df_.ECI < 8
                )

            [['TagTime', 'ZTPOK', 'OgCHP', 'OgCoal', 'BB', 'ECI', 'ECII','turnOff']]
            )

        return  dfReturn

    def createOperationBooleanList(self, dataframe):
        """
        Function which creates list witch boolean values which indicates if unit ECI is defined as working by optimizer.
        Also it return dataframe with optimizer results.

        :param dataframe: dataframe with data prepared for optimization by prepareDataForOptimization function.
        :return: rows - list with boolean values indicates if ECI is working or not,
                df - dataframe with optimizer results
        """

        rows = []
        df = self.runOptimizer(dataframe)
        for idx in range(len(df)):

            valuesDict = {f'value{key}': 0  for key in range(self.windowSize)}

            for keyIdx, key in enumerate(valuesDict.keys()):

                try:
                    valuesDict[key] = df.turnOff.iloc[idx + keyIdx]
                except:
                    valuesDict[key] = True

            rows.append(list(valuesDict.values()))

        return rows, df

    def getSchedule(self, dataframe):
        """
        Function which check if ECI is working within defined conditions, it means it is operating and not working at
        least within defined windowSize.
        If needed ECI is stopped and it's power defined by optimizer is generated by ECII and vice versa - ECI is turn
        on and it's power is substracted from ECII power.
        .
        :param dataframe: dataframe with data prepared for optimization by prepareDataForOptimization function.
        :return: dfTest - dataframe with defined schedule.
        """

        newRows = []
        idx = 0
        df = dataframe.copy()
        rows, dfTest = self.createOperationBooleanList(df)

        try:
            while idx <= len(dfTest):
                if len(newRows) == len(dfTest):
                    break

                values = dfTest.turnOff.iloc[idx:(idx + self.windowSize)].values

                if False in values:
                    indexes = []
                    for index,value in enumerate(values):
                        if value == False:
                            indexes.append(index)

                    if 0 in indexes:
                        if len(indexes) >= self.windowSize :
                            for _ in range(len(indexes)):
                                newRows.append(False)
                                idx += 1

                        elif ((idx >= self.windowSize) and (rows[idx - self.windowSize+1].count(False) >= self.windowSize)):
                            newRows.append(False)
                            idx += 1

                        elif ((idx + self.windowSize <= len(dfTest)) and (rows[idx + self.windowSize-1].count(False) >= self.windowSize)):

                            for _ in range(len(indexes)):
                                newRows.append(False)
                                idx += 1
                        else:
                            newRows.append(True)
                            idx += 1

                    else:

                        if ((len(indexes) >= (self.windowSize/2)) and (newRows[idx-1] == False)):
                            newRows.append(False)
                            idx += 1

                        elif ((newRows[idx-1] == False) and (indexes[0] == (self.windowSize-1))):
                            for _ in range(self.windowSize):
                                newRows.append(False)
                                idx += 1
                        else:
                            newRows.append(True)
                            idx += 1

                else:
                    #If there is no operation hours within the windowSize just keep it not working
                    newRows.append(True)
                    idx += 1

        except Exception as e:
            print(f'Error {e} for index {idx}')

        dfTest = (dfTest
                  .assign(
                      turnOff = pd.Series(newRows),

                      ECII = lambda df_: np.where(df_.turnOff == False,
                                                np.where(df_.ECI == 0,
                                                         df_.ECII - self.lowerBoundaryPowerECI,
                                                         df_.ECII ),
                                                df_.ECII + df_.ECI),
                      ECI = lambda df_: np.where(df_.turnOff == False,
                                                np.where(df_.ECI == 0,
                                                         self.lowerBoundaryPowerECI,
                                                         df_.ECI ),
                                                0),
                  )
                  [['TagTime','ZTPOK', 'OgCHP', 'OgCoal', 'BB','ECI','ECII']]
                 )


        return dfTest

class DBIteraction():
    def __init__(self, serverDict):
        self.serverDict = serverDict

        self.conn = pyodbc.connect(
                                    f'Driver={self.serverDict["Driver"]};'
                                    f'Server={self.serverDict["Server"]};'
                                    f'Database={self.serverDict["Database"]};'
                                    'Trusted_Connection=yes;'
                                    )



    def deteleExistingSchedule(self):

        cursor = self.conn.cursor()

        query = (f"""
        DELETE FROM {self.serverDict["Database"]}.[dbo].{self.serverDict["Table"]}
        """)

        cursor.execute(query)

        self.conn.commit()

        # Finish of cursor operation
        cursor.close()

    def saveNewSchedule(self, result):

        cursor = self.conn.cursor()

        # Iterate through dataframe rows
        try:
            for row in result.values:

                #Query
                query = (f"""
                    INSERT INTO dbo.{self.serverDict["Table"]}([TagTime],[ZTPOK],[OgCHP],[OgCoal],[BB],[ECI],[ECII])
                    VALUES (?,?,?,?,?,?,?)
                        """)

                # Run query with defined input
                cursor.execute(query, tuple(row))

                # Commit changes in the database
                self.conn.commit()

        except Exception as error:
            print(f" Błąd zapisu danych do bazy: {error}")
            print()


        # Finish of cursor operation
        cursor.close()


    def loadForecasts(self):

        cursor = self.conn.cursor()

        query = (f"""
                SELECT [TagTime]
                      ,[PrognozaObciazeniaMiasto]
                      ,[PrognozaObciazeniaFordon]
                      ,[PrognozaTemperatury]
                FROM {self.serverDict["Database"]}.[dbo].{self.serverDict["View"]}
                """)

        cursor.execute(query)
        # Fetch all rows from the executed query
        rows = cursor.fetchall()

        # convert the tuple of tuples to a list of dictionaries
        list_of_dicts = [{'TagTime': tagtime,
                          'PrognozaObciazeniaMiasto': miastoLoad,
                          'PrognozaObciazeniaFordon': fordonLoad,
                          'PrognozaTemperatury': temperature} for tagtime, miastoLoad, fordonLoad, temperature in rows]

        # convert the list of dictionaries to a Pandas DataFrame
        dataframe = pd.DataFrame(list_of_dicts)


        self.conn.commit()

        # Finish of cursor operation
        cursor.close()

        return dataframe



    def closeConnection(self):
        self.conn.close()


def main():
    """
    Function which run whole optimization of production schedule process

    :return: dataframe with schedule for all units defined.
    """

    print("Starting schedule definition")
    serverDict = {
        'Driver': 'SQL Server',
        'Server': 'KPEC-KELVIN',
        'Database': 'BAZA_INTEGRACYJNA',
        'View': 'v_PrognozyHarmonogramProdukcji',
        'Table': 'tbl_HarmonogramProdukcji'
    }

    dbObj = DBIteraction(serverDict)
    print("Lading data from database")
    forecastsDf = dbObj.loadForecasts()
    objInstance = Scheduler()

    print("Data transformation")
    dataframe = objInstance.prepareDataForOptimization(forecastsDf)

    print("Schedule definition")
    result = objInstance.getSchedule(dataframe)

    print("Deleting existing data in database")
    dbObj.deteleExistingSchedule()

    print("Saving new schedule into database")
    dbObj.saveNewSchedule(result.round(2))
    dbObj.closeConnection()

    print("Schedule definition finished")
    return print(result.round(2))


if __name__ == "__main__":
    main()
