function [p, y, t] = import_data_load(split)
%IMPORT_DATA_LOAD Hourly Panama electricity demand in the NARX [p, y] format.
%
%   [P, Y] = IMPORT_DATA_LOAD() returns the external input p(t) and the
%   measured target y(t) of the default training split as column vectors in
%   their recorded units, ready for NARXmodel.train(p, y, train_k):
%
%     P   air temperature at Tocumen, column T2M_toc of the continuous
%         dataset, in degrees Celsius. It is the external input p(t).
%     Y   national hourly electricity demand, column nat_demand, in MW. It
%         is the first data column of the continuous dataset and the
%         measured target y(t).
%
%   [P, Y, T] = IMPORT_DATA_LOAD(...) also returns the hourly datetime
%   index, so a caller can group by hour of day, weekday or season without
%   re-reading the file.
%
%   SPLIT selects a contiguous span of the hourly series:
%
%     'train'      2015-01-03 01:00 to 2018-12-31 23:00  (35,015 hours)
%     'test'       2019-01-01 00:00 to 2019-12-31 23:00  ( 8,760 hours)
%     'dry-train'  2017-12-15 00:00 to 2018-04-15 23:00  ( 2,928 hours)
%     'dry-val'    2016-12-15 00:00 to 2017-04-15 23:00  ( 2,928 hours)
%     'dry-test'   2019-01-01 00:00 to 2019-04-15 23:00  ( 2,520 hours)
%     'all'        the complete file, to 2020-06-27 00:00 (48,048 hours)
%     [t1 t2]      any datetime range, for example one rainy season:
%                  import_data_load([datetime(2018,5,1), datetime(2018,11,30)])
%
%   'dry-train', 'dry-val' and 'dry-test' are the three windows the delay
%   experiment uses. Each is one Panamanian dry season (mid-December to
%   mid-April) of a different year: one to fit on, one to choose between
%   fitted networks on, and one, held out until the end, to report on.
%   Fitting one model per season and applying it out of year without
%   refitting is the protocol of Hagan's short-term load forecasting work. Note that the
%   seasonal swing in this record is small -- monthly mean demand spans
%   1148.6 to 1205.6 MW and the mean daily peak-to-trough amplitude spans
%   455.4 to 496.6 MW -- so a season window is mainly a homogeneous,
%   affordable span, not a way of isolating a large annual cycle.
%
%   All four named splits end before 2020-03-16, when COVID-19
%   restrictions moved the monthly mean demand down by about 15%
%   (2020-02: 1269.5 MW, 2020-04: 1062.0 MW). That regime shift is only
%   reachable through 'all'.
%
%   The returned span is always contiguous and hourly, which is what the
%   tapped delay lines of the NARX model require: a delay of 1 is one hour,
%   24 is one day and 168 is one week. The function asserts this rather
%   than assuming it.
%
%   The recorded series is returned unaltered. It contains a small number
%   of genuine outage hours (13 hours below 700 MW, the lowest 85.2 MW on
%   2019-01-20), which are measurements of the grid, not missing data.
%
%   See also TRAIN_NARX_LOAD, SAMPLE_LOAD_WINDOWS, NARXMODEL.

    if nargin < 1
        split = 'train';
    end

    dataFile = fullfile(fileparts(mfilename('fullpath')), 'continuous dataset.csv');
    data = readtable(dataFile);

    t = data.datetime;
    p = data.T2M_toc;
    y = data.nat_demand;

    span = split_span(split);
    keep = t >= span(1) & t <= span(2);
    t = t(keep);
    p = p(keep);
    y = y(keep);

    assert(~isempty(t), 'The requested span selects no hours.');
    assert(all(diff(t) == hours(1)), ...
        'The selected span is not a continuous hourly series.');

end

function span = split_span(split)
%SPLIT_SPAN Map a named split, or pass a caller-supplied datetime range.

    if isdatetime(split)
        assert(numel(split) == 2, 'A datetime split must be [t1 t2].');
        span = split;
        return;
    end

    switch lower(split)
        case 'train'
            span = [datetime(2015,1,3,1,0,0), datetime(2018,12,31,23,0,0)];
        case 'test'
            span = [datetime(2019,1,1,0,0,0), datetime(2019,12,31,23,0,0)];
        case 'dry-train'
            span = [datetime(2017,12,15,0,0,0), datetime(2018,4,15,23,0,0)];
        case 'dry-val'
            span = [datetime(2016,12,15,0,0,0), datetime(2017,4,15,23,0,0)];
        case 'dry-test'
            span = [datetime(2019,1,1,0,0,0), datetime(2019,4,15,23,0,0)];
        case 'all'
            span = [datetime(2015,1,3,1,0,0), datetime(2020,6,27,0,0,0)];
        otherwise
            error(['Unknown split ''%s''. Use train, test, dry-train, ', ...
                   'dry-val, dry-test, all or [t1 t2].'], split);
    end

end
