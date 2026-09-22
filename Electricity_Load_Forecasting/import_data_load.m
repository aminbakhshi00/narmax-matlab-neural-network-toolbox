function [p, y, t, segmentStart] = import_data_load(split)
%IMPORT_DATA_LOAD Hourly Panama electricity demand in the NARX [p, y] format.
%
%   [P, Y] = IMPORT_DATA_LOAD(SPLIT) returns the external input p(t) and the
%   measured target y(t) as column vectors in their recorded units, ready
%   for NARXmodel.train(p, y, train_k):
%
%     P   air temperature at Tocumen, column T2M_toc, in degrees Celsius.
%     Y   national hourly electricity demand, column nat_demand, in MW.
%
%   [P, Y, T] = IMPORT_DATA_LOAD(...) also returns the hourly datetime index.
%
%   [P, Y, T, SEGMENTSTART] = IMPORT_DATA_LOAD(...) also returns the row
%   index at which each continuous stretch begins. A split made of adjacent
%   files is one stretch, so SEGMENTSTART is 1; a split that joins seasons
%   from different years has one entry per stretch, and the caller can use
%   them to avoid building a window that straddles a join.
%
%   The record is stored as one CSV per season under data/seasons, cut so
%   that every file is internally continuous and hourly. This function only
%   reads the files a split names and concatenates them; all of the cutting
%   was done once, up front. data/raw holds the original download untouched,
%   and data/seasons/manifest.csv lists every segment with its span.
%
%   SPLIT is one of:
%
%     'dry-train'  the 2016-17 and 2017-18 dry seasons   (5,856 hours)
%     'dry-val'    the 2015-16 dry season                (2,952 hours)
%     'dry-test'   1 Jan to 15 Apr 2019, held out        (2,520 hours)
%     'train'      2015-01-03 to 2018-12-31             (35,015 hours)
%     'test'       calendar 2019                         (8,760 hours)
%     'all'        everything, including the COVID-19 period (48,048 hours)
%
%   or a string array of segment names, for example
%   ["dry_2016-2017", "dry_2017-2018"].
%
%   'dry-train' is two dry seasons rather than one: fitting and testing on
%   the same kind of season is what keeps the model in one regime, and the
%   2015-16 season -- previously unused -- frees 2016-17 to join the
%   training set. The two are a year apart, so the series handed to the
%   model has one join in it, which is what SEGMENTSTART marks.
%
%   Each named split is one Panamanian dry season (mid-December to
%   mid-April) or a span built from whole seasons. Fitting on dry seasons
%   and applying out of year without refitting is the protocol of Hagan's
%   short-term load forecasting work.
%
%   See also SAMPLE_LOAD_WINDOWS, NARXMODEL.

if nargin < 1
    split = 'dry-train';
end

dry = ["dry_2014-2015", "wet_2015", "dry_2015-2016", "wet_2016", ...
       "dry_2016-2017", "wet_2017", "dry_2017-2018", "wet_2018", ...
       "dry_2018-2019_dec", "dry_2018-2019_janapr", "wet_2019", ...
       "dry_2019-2020_dec", "dry_2019-2020_janmar", "covid_2020"];

if isstring(split) || iscellstr(split) %#ok<ISCLSTR>
    files = string(split);
else
    switch lower(string(split))
        case "dry-train", files = ["dry_2016-2017", "dry_2017-2018"];
        case "dry-val",   files = "dry_2015-2016";
        case "dry-test",  files = "dry_2018-2019_janapr";
        case "train",     files = dry(1:9);
        case "test",      files = ["dry_2018-2019_janapr", "wet_2019", ...
                                   "dry_2019-2020_dec"];
        case "all",       files = dry;
        otherwise
            error('import_data_load:unknownSplit', ...
                'Unknown split "%s". See HELP IMPORT_DATA_LOAD.', string(split));
    end
end

seasonFolder = fullfile(fileparts(mfilename('fullpath')), 'data', 'seasons');
blocks = cell(1, numel(files));
for k = 1:numel(files)
    blocks{k} = readtable(fullfile(seasonFolder, files(k) + ".csv"));
end
data = vertcat(blocks{:});

t = datetime(data.datetime);
p = data.T2M_toc;
y = data.nat_demand;

% A join is any place the hourly index skips. Files that happen to be
% adjacent in time therefore read as one continuous stretch.
segmentStart = [1; find(diff(t) ~= hours(1)) + 1];
end
