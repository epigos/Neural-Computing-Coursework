function DataAnalysis(data)
    fprintf("Performing exploratory data analysis...\n");
    
    % Data Shape
    [rows, columns] = size(data);
    % define categorical column parameters
    catColumns = {'Family', 'Genus', 'Species'};
    
    fprintf("The data contains %d observations with %d columns\n\n", rows, columns);
    fprintf("The data set contains %d categorical and %d numerical features.\n",...
        length(catColumns), columns - length(catColumns));
    % show first 5 rows
    disp("Display the first 5 rows");
    head(data, 5)
    % summarize the data
    disp("Print summary of data table:");
    summary(data);
end

