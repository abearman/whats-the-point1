numClassesPerImage = 1.5;
numInstancesPerImage = 2.8;
numPASCAL = 12031;

timePerLabel = 1;
timePerClick = 2.4;
timePerSquiggle = 10.9; 
timePerBox = 26;
timePerSegmentation = 79;

imgIOU = 32.2;
clickIOU = 42.7;
clickIOUtest = 43.6;
squiggleIOU = 49.1;
boxIOU = 45.1;
fsIOU = 58.3;

imgTime = 20*timePerLabel;
clickTime = (20-numClassesPerImage)*timePerLabel +...
        numClassesPerImage*timePerClick;
squiggleTime = (20-numClassesPerImage)*timePerLabel +...
	numClassesPerImage*timePerSquiggle;
boxTime = (20-numClassesPerImage)*timePerLabel +...
        numInstancesPerImage*timePerBox;
segmTime = (20-numClassesPerImage)*timePerLabel +...
        numInstancesPerImage*timePerSegmentation;

fontSize = 24*0.9;
fontSizeClick = 24*1.1;

markerSize = 40;
markerSizeClick = 40*1.2;

