function DataAnalysis(data)
    fprintf("Performing exploratory data analysis...\n");
    
    % Data Shape
    [rows, columns] = size(data);
    % define categorical column parameters
    catColumns = {'Sex'};
    numericCols = {'Length', 'Diameter', 'Height', 'Whole weight',...
        'Shucked weight', 'Viscera weight', 'Shell weight'};
    
    fprintf("The data contains %d observations with %d columns\n\n", rows, columns);
    fprintf("The data set contains %d categorical and %d numerical features.\n",...
        length(catColumns), length(numericCols));
    % show first 5 rows
    disp("Display the first 5 rows");
    head(data, 5)
    % summarize the data
    disp("Print summary of data table:");
    summary(data);
end

