# -*- coding: utf-8 -*-
'''
Site

A site import and analysis class built
with the pandas library
'''
import anemoi as an
import pandas as pd
import numpy as np
import itertools

class Site(object):
    '''Subclass of the pandas dataframe built to import and quickly analyze
       met mast data.'''

    def __init__(self, masts=None, meta_data=None, primary_mast=None):
        '''Data structure with an array of anemoi.MetMasts and a DataFrame of
        results:

        Parameters
        ----------
        masts: array of anemoi.MetMasts
        meta_data: DataFrame of analysis results
        primary_mast: string or int, default None
            Longest-term mast installed on site
        '''

        if masts is not None:
            mast_names = []
            mast_lats = []
            mast_lons = []
            mast_heights = []
            mast_primary_anos = []
            mast_primary_vanes = []

            for mast in masts:
                if isinstance(mast, an.MetMast):
                    mast_names.append(mast.name)
                    mast_lats.append(mast.lat)
                    mast_lons.append(mast.lon)
                    mast_heights.append(mast.height)
                    mast_primary_anos.append(mast.primary_ano)
                    mast_primary_vanes.append(mast.primary_vane)

        if meta_data is None:
            meta_data = pd.DataFrame(columns=mast_names,
              index=['Lat', 'Lon', 'Height', 'PrimaryAno', 'PrimaryVane'])
            meta_data.loc['Lat', :] = mast_lats
            meta_data.loc['Lon', :] = mast_lons
            meta_data.loc['Height', :] = mast_heights
            meta_data.loc['PrimaryAno', :] = mast_primary_anos
            meta_data.loc['PrimaryVane', :] = mast_primary_vanes

        meta_data.columns.name = 'Masts'
        self.masts = masts
        self.meta_data = meta_data

    def __repr__(self):
        mast_names = 'Site masts: '
        for mast in self.masts:
            if mast_names == 'Site masts: ':
                mast_names = mast_names + ' ' + str(mast.name)
            else:
                mast_names = mast_names + ', ' + str(mast.name)
        return mast_names


    def check_has_masts(self):
        if len(self.masts) < 1:
            raise ValueError("This site doesn't seem to have any masts associated...")

        return True

    def get_mast_names(self):
        if not self.masts:
            raise ValueError("This site doesn't seem to have any masts associated...")
        else:
            return self.meta_data.columns

    def return_ws_corr_results_binned_by_direction(self):
        if self.check_has_masts():
            site_correlation_results = []

            for mast_pair in itertools.permutations(self.masts, 2):
                ref_mast = mast_pair[0]
                site_mast = mast_pair[1]
                results = an.correlate.correlate_masts_10_minute_by_direction(ref_mast=ref_mast, site_mast=site_mast)
                site_correlation_results.append(results)

            site_correlation_results = pd.concat(site_correlation_results, axis=0)
            return site_correlation_results

    def return_cross_corr_results_dataframe(self):
        if self.check_has_masts():
            cross_corr_results_index = pd.MultiIndex.from_product([self.meta_data.columns.tolist()]*2, names=['Ref', 'Site'])
            results_cols = ['Slope', 'Offset', 'DirOffset', 'R2', 'Uncert']
            cross_corr_results_dataframe = pd.DataFrame(index=cross_corr_results_index, columns=results_cols)
            refs = cross_corr_results_dataframe.index.get_level_values(level='Ref')
            sites = cross_corr_results_dataframe.index.get_level_values(level='Site')
            cross_corr_results_dataframe = cross_corr_results_dataframe.loc[refs != sites, :]
            return cross_corr_results_dataframe

    def calculate_measured_momm(self):
        '''Calculates measured mean of monthly mean wind speed for each mast in anemoi.Site'''
        if self.check_has_masts():
            for mast in self.masts.Masts:
                self.meta_data.loc['Meas MoMM', mast.name] = mast.return_momm(sensors=mast.primary_ano).iloc[0,0]

    def calculate_self_corr_results(self):
        if self.check_has_masts():
            cross_corr_results = self.return_cross_corr_results_dataframe()
        for mast_pair in cross_corr_results.index:
            ref = mast_pair[0]
            site = mast_pair[1]
            ref_mast = self.masts.loc[ref,'Masts']
            site_mast = self.masts.loc[site,'Masts']

            slope, offset, uncert, R2 = site_mast.correlate_to_reference(reference_mast=ref_mast, method='ODR')
            results_cols = ['Slope', 'Offset', 'R2', 'Uncert']
            cross_corr_results.loc[pd.IndexSlice[ref, site], results_cols] = [slope, offset, R2, uncert]

        return cross_corr_results

    def calculate_annual_shear_results(self):
        if self.check_has_masts():
            shear_results = an.shear.shear_analysis_site(self.masts)
            return shear_results


    def calculate_long_term_alpha(self):
        '''Calculates measured annual alpha for each mast in anemoi.Site'''
        if self.check_has_masts():
            for mast in self.masts:
                self.meta_data.loc['Alpha', mast.name] = mast.calculate_long_term_alpha()

    def plot_monthly_valid_recovery(self):
        '''Plots monthly valid recovery for each mast in anemoi.Site'''
        if self.check_has_masts():
            for mast in self.masts:
                mast.plot_monthly_valid_recovery()

    def plot_freq_dists(self):
        '''Plots wind speed frequency distributions for each mast in anemoi.Site'''
        if self.check_has_masts():
            for mast in self.masts:
                mast.plot_freq_dist()

    def plot_wind_roses(self):
        '''Plots wind speed frequency distributions for each mast in anemoi.Site'''
        if self.check_has_masts():
            for mast in self.masts:
                mast.plot_wind_rose()

    def plot_site_masts_summary(self):
        for mast in self.masts:
            print(mast.mast_data_summary(), mast, '\n')
            mast.plot_monthly_valid_recovery();
            mast.plot_wind_energy_roses(dir_sectors=12);
            mast.plot_freq_dist();
            # plt.show()

    def plot_ws_corr_results_binned_by_direction(self):
        site_correlation_results = self.return_ws_corr_results_binned_by_direction()

        dir_bins = site_correlation_results.index.get_level_values('DirBin').unique()

        for mast_pair in itertools.permutations(self.masts, 2):
            ref_mast = mast_pair[0]
            ref_mast_name = ref_mast.name
            site_mast = mast_pair[1]
            site_mast_name = site_mast.name

            ref_data = ref_mast.return_sensor_data([ref_mast.primary_ano, ref_mast.primary_vane])
            site_data = site_mast.return_sensor_data(site_mast.primary_ano)
            df = pd.concat([ref_data, site_data], axis=1, join='inner', keys=['Ref', 'Site']).dropna()
            df.columns = ['RefWS', 'RefDir', 'SiteWS']
            df = an.correlate.append_dir_bin(df, dir_column='RefDir')

            an.plotting.plot_ws_correlation_by_direction(df=df,
                                                        site_corr_results=site_correlation_results,
                                                        site_mast_name=site_mast_name,
                                                        ref_mast_name=ref_mast_name)
