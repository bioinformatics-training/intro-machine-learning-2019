count = 0;
files = dir('./Rick1/*.jpg');

% Loop through each
for id = 1:length(files)
    count = count+1;
    % Get the file name (minus the extension)
    [~, f] = fileparts(files(id).name);
      % Convert to number
      %num = str2double(f);
      if ~isnan(count)
          % If numeric, rename
          copyfile(['./Rick1/' files(id).name], ['./Rick/Rick_' sprintf('%03d.jpg', count)]);
      end
end


count = 0;
files = dir(['./Morty1/*.jpg']);

%files = dir('../../CrossValidation/Rick/*.jpg');
% Loop through each
for id = 1:length(files)
    count = count+1;
    % Get the file name (minus the extension)
    [~, f] = fileparts(files(id).name);
      % Convert to number
      %num = str2double(f);
      if ~isnan(count)
          % If numeric, rename
          copyfile(['./Morty1/' files(id).name], ['./Morty/Morty_' sprintf('%03d.jpg', count)]);
      end
end