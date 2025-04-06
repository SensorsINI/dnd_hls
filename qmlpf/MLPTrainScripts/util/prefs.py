# from __future__ import print_function
from typing import Any

from pypref import SinglePreferences

class MyPreferences(SinglePreferences):
    """ Class to hold user preference values like java.util.Preferences; adds put() method to pypref SinglePreferences.
    The preferences are stored (by default) as 
    
    To use it, put these lines at start of your file

        from prefs import MyPreferences # or whereever you put prefs.py
        prefs=MyPreferences() # store and retrieve sticky values

    Later in code
        self.initial_position_slider.setValue(prefs.get('initial-position',0))  # get vqlue for initial position, or default of 0 if not in prefs file (setup)

        prefs.put('initial-position',value) # store a value for 'initial-position'

    """
    # *args and **kwargs can be replaced by fixed arguments
    def put(self,key:str, value:Any):
        """ Put a value to preferences using key
        :param key: a string key
        :param value: the value to put
        """
        self.update_preferences({key:value})