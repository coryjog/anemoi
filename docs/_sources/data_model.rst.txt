Data Model
==========

This will be a quick explination of how this library is structured so that analysts have a clear understanding of how met mast data are manipulated and analized.

Anemoi MetMast object 
----------------------

This is the foundational object within the Anemoi wind analysis package and upon which the rest of the library is built. This is the equivalent of the DataFrame to pandas. The MetMast object is made up of two parts:

1. Data - A time series of measured wind data within a pandas DataFrame. The DataFrame is indexed by the time stamps of the time series and the columns are labeled with the sensor names. Normally this includes wind speed, direction, and temperature. It can also include pressure, relative humidity, battery voltage, and any other signals recorded by the data logger.
2. Metadata - this is pertinent mast information such as coordinates, height, elevation, primary anemometer and wind vane. Other information could be added in the future.

Once a MetMast object is created, the analyst can access the data and/or the metadata using the following:

.. code-block:: python
    :linenos:

    import anemoi as an
    
    # create MetMast object named mast
    mast = an.MetMast()
    
    # print data and metadata
    print(mast.data)
    print(mast.metadata)

For a bit more information on the MetMast object, you can see a `presentation here <http://slides.com/coryjog/anemoi-plan/fullscreen?token=bI4mJCcM>`_.

Sensor naming convention
-------------------------

The actual measured mast data within an.MetMast.data has some additional restrictions around sensor naming. Sensor naming must follow EDF's convention. This is so that anemoi can easily extract needed type, height, orientation, and signal from the sensor name. Mast data are organized by these sensor attributes so the sensor name includes these important values. 

Sensor names have the following format: type_height_orientation_signal

* Type: This is the sensor type so that anemoi knows the difference between an anemometer and a wind vane. SPD, DIR, T, BP, RH, VBAT are all valid sensor types. These labels correspond to anemometer, wind vane, thermometer, pressure, relative humidity, and battery voltage.
* Height: This is the installed height, in meters, of the sensor.
* Orientation: This is the cardinal direction of the sensor. N, NE, E, SE, S, SW, W, NW are all valid orientations. 
* Signal: This is the signal type of the column. AVG, SD, MIN, MAX are all valid signal types. These labels correspond to average, standard deviation, minimum and maximum. If the sensor doesn't have a signal type in the name then average is assumed.

Sensor name examples
-------------------------

Examples:

.. code-block:: python
    :linenos:

    'SPD_58_SW'     # Average wind speed from a southwest oriented anemometer at 58 m 
    'SPD_58_SW_SD'  # Standard deviation of wind speed from a southwest oriented anemometer at 58 m
    'SPD_58_SW_MAX' # Maximum wind speed from a southwest oriented anemometer at 58 m

    'SPD_32.2_N'    # Average wind speed from a north oriented anemometer at 32.2 m 
    'SPD_32.2_N_SD' # Standard deviation of wind speed from a north oriented anemometer at 32.2 m
    'SPD_32.2_N_MAX'# Maximum wind speed from a north oriented anemometer at 32.2 m

    'DIR_80'        # Average wind direction from a wind vane at 80 m 
    'SPD_80__SD'    # Standard deviation of wind direction from a wind vane 80 m

    'T_3'           # Average temperature from a thermometer at 3 m 
    'T_3__SD'       # Standard deviation temperature from a thermometer at 3 m 


Orientations of wind vanes and thermometers aren't necessarily required for an analysis although the two underscores do need to be included so that the signal type is aligned with the names of sensors where orientation is important.


