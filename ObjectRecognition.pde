import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Vector;
import java.math.BigInteger;

//String strDistanceMetric = "minExpDistance";
String strDistanceMetric = "bhattacharyyaDistance";
//String strDistanceMetric = "KL_Divergence";
//String strDistanceMetric = "jensenShannonDivergence";

boolean isFeaturesToTextFile  = true;

int blockSize = 15; //DCT Block-Size for Image Enhancement

String outputType = "DCTEnhancedImage";
//String outputType = "originalImage";

//Aspect ratio is 4:3
int numOfColumns = 8;
int numOfRows    = 6;

String trainFilename = "TrainData.arff";
File TrainFile;

String testFilename  = "TestData.arff";
File TestFile;

String database = "./train/train";//"FlickrFaceDatabase";     //NumOfClasses = 30
//String database = "FlickrFaceDatabase2";  //NumOfClasses = 208
//String database = "FlickrFaceDatabase3";  //NumOfClasses = 208
//String database = "MMU1IrisDatabase";     //NumOfClasses = 45
//String database = "MMU2IrisDatabase";     //NumOfClasses = 100

int numOfClasses = 30;
float trainDBRatio = 0.70F;
int histLength = 8;

BigInteger seed1 = BigInteger.valueOf( 0 );
BigInteger seed2 = BigInteger.valueOf( 0 );

int totalBlocks = numOfColumns * numOfRows;

void setup()
{
  if( isFeaturesToTextFile )
  {
    TrainFile = new File( dataPath( trainFilename ) );
    
    if( TrainFile.exists() && TrainFile.isFile() )
    {
      TrainFile.delete();
    }
    
    createFile( TrainFile );
    
    TestFile = new File( dataPath( testFilename ) );
    
    if( TestFile.exists() && TestFile.isFile() )
    {
      TestFile.delete();
    }
    
    createFile( TestFile );
  }
  
  ArrayList objectDatabase = new ArrayList();
  ArrayList nameDatabase = new ArrayList();
  
  trainDatabase( objectDatabase, nameDatabase, 1, numOfClasses );
  testDatabase( objectDatabase, nameDatabase, 1, numOfClasses );
}

float [] getFeatures( PImage img, String path )
{ 
  int adjustedWidth  = img.width;
  int adjustedHeight = img.height;
  
  // Adjust width to the nearest value that is divisible by numOfColumns
  int lowerWidth = adjustedWidth;
  while (lowerWidth % numOfColumns != 0)
  {
    lowerWidth--;
  }
    
  int upperWidth = adjustedWidth;
  while (upperWidth % numOfColumns != 0)
  {
    upperWidth++;
  }

  // Choose the nearest width
  adjustedWidth = (adjustedWidth - lowerWidth < upperWidth - adjustedWidth) ? lowerWidth : upperWidth;

  // Adjust height to the nearest value that is divisible by numOfRows
  int lowerHeight = adjustedHeight;
  while (lowerHeight % numOfRows != 0)
  {
    lowerHeight--;
  }
    
  int upperHeight = adjustedHeight;
  while (upperHeight % numOfRows != 0)
  {
    upperHeight++;
  }

  // Choose the nearest height
  adjustedHeight = (adjustedHeight - lowerHeight < upperHeight - adjustedHeight) ? lowerHeight : upperHeight;
  
  img.resize( adjustedWidth, adjustedHeight );
  img.updatePixels();
  
  if( outputType == "DCTEnhancedImage" )
    img = getDCTEnhancedImage( img );
    
  img.save("output/" + path);
  
  ArrayList histList = new ArrayList();
  
  int blockWidth  = img.width / numOfColumns;
  int blockHeight = img.height / numOfRows;
        
  // Calculate the total number of blocks
  totalBlocks = 0;
  
  for( int y = 0; y < img.height; y += blockHeight )
  {
    for( int x = 0; x < img.width; x += blockWidth )
    {
      totalBlocks++;
      
      // Blok boyutlarını belirle
      int w = min(blockWidth, img.width - x);
      int h = min(blockHeight, img.height - y);
      
      // Bloku kırp ve ekrana çiz
      PImage block = img.get(x, y, w, h);
      
      float [] hist = new float[histLength];
      
      for( int i = 0; i < block.pixels.length; i++ )
      {
        hist[ (int) ( brightness( block.pixels[i] ) / ( 256 / histLength ) ) ]++;
      }
      
      for( int i = 0; i < histLength; i++ )
      {
        hist[i] /= block.pixels.length;
        histList.add( hist[i] );
      }
      
    }
  }
  
  float [] histArr = new float[histList.size()];
  
  float sum = 0F;
  
  for( int i = 0; i < histArr.length; i++ )
  {
    histArr[i] = (float) ( (Float) histList.get( i ) );
    sum += histArr[i];
  }
  
  for( int i = 0; i < histArr.length; i++ )
  {
    histArr[i] /= sum;
  }
    
  return histArr;
}

String [] readImagesPath( String dir )
{
  File folder = new File( dir );
  File [] listOfFiles = folder.listFiles();
  
  String [] imgNames = null;
  
  if( listOfFiles != null && listOfFiles.length > 0 )
  {
    imgNames = new String[listOfFiles.length];
    
    int imgCounter = 0;
    
    for( File file : listOfFiles )
    {
      if( file.isFile() && file.exists() )
      {
        imgNames[imgCounter] = file.getName();
        imgCounter++;
      }
      else
      {
        println( "File is null!" );
      }
    }
  }
  else
  {
    println( "Folder is empty!" );
  }
  
  return imgNames;
}

void trainDatabase( ArrayList objectDatabase, ArrayList nameDatabase, int start, int end )
{ 
  println( "-------------------------------------------" );
  println( "Training started!" );
  
  int numOfFeatures = totalBlocks * histLength;
        
  String header = "@RELATION objectRecognition\n";
          
  for( int s = 1; s <= numOfFeatures; s++ )
    header += "@ATTRIBUTE A" + s + " NUMERIC\n";
  
  header += "@ATTRIBUTE class {";
  
  for( int r = 1; r <= numOfClasses; r++ )
    header += r + ", ";
  
  header += "}\n@DATA\n";
  
  appendTextToFile( TrainFile, header );
  
  for( int object = start; object <= end; object++ )
  {
    String [] imgNames = readImagesPath( sketchPath( "data/" + database + "/" + object + "/" ) );
    
    if( imgNames != null && imgNames.length > 0 )
    {
      shuffleArray( imgNames, 1 );
      
      for( int ii = 0; ii < floor( imgNames.length * trainDBRatio ); ii++ )
      {
        PImage img = loadImage( database + "/" + object + "/" + imgNames[ii] );
        
        if( img != null && img.width > 0 && img.height > 0 )
        {
          img.loadPixels();
          
          float [] pdfVector = getFeatures( img, database + "/train/" + object + "/" + imgNames[ii] );
          
          if( pdfVector != null && pdfVector.length > 0 )
          {
            String str = "";
            
            for( int s = 0; s < pdfVector.length; s++ )
            str += pdfVector[s] + ",";
            
            
            if( isFeaturesToTextFile )
            {
              str += String.valueOf( object );
              appendTextToFile( TrainFile, str );
            }
            
            objectDatabase.add( pdfVector );
            nameDatabase.add( String.valueOf( object ) );
            
            println( imgNames[ii] + " was registered to database." );
          }
        }
      }
    }
  }
  
  println( "-------------------------------------------" );
  println( "Training finished!" );
  println( "-------------------------------------------" );
}

void testDatabase( ArrayList objectDatabase, ArrayList nameDatabase, int start, int end )
{
  long startTime = System.currentTimeMillis();
  
  println( "Testing started!" );
  
  int numOfFeatures = totalBlocks * histLength;
  
  String header = "@RELATION objectRecognition\n";
          
  for( int s = 1; s <= numOfFeatures; s++ )
    header += "@ATTRIBUTE A" + s + " NUMERIC\n";
  
  header += "@ATTRIBUTE class {";
  
  for( int r = 1; r <= numOfClasses; r++ )
    header += r + ", ";
  
  header += "}\n@DATA\n";
  
  appendTextToFile( TestFile, header );
  
  int numOfTrueDetections  = 0;
  int numOfFalseDetections = 0;
  
  int counter = 0;
  
  float accuracy = 0.0f;
  
  for( int object = start; object <= end; object++ )
  {
    String [] imgNames = readImagesPath( sketchPath( "data/" + database + "/" + object + "/" ) );
    
    if( imgNames != null && imgNames.length > 0 )
    {
      shuffleArray( imgNames, 2 );
      
      for( int ii = imgNames.length - 1; ii >= floor( imgNames.length * trainDBRatio ); ii-- )
      {
        PImage img = loadImage( database + "/" + object + "/" + imgNames[ii] );
        
        if( img != null && img.width > 0 && img.height > 0 )
        {
          img.loadPixels();
          
          if( ( objectDatabase.size() > 0 ) && ( nameDatabase.size() > 0 ) )
          {
            float [] pdfVector = getFeatures( img, database + "/test/" + object + "/" + imgNames[ii] );
            
            String str = "";
            
            if( pdfVector != null && pdfVector.length > 0 )
            {
              for( int s = 0; s < pdfVector.length; s++ )
                str += pdfVector[s] + ",";
            
              if( isFeaturesToTextFile )
              {
                str += String.valueOf( object );
                appendTextToFile( TestFile, str );
              }
              
              String [] result = compare( pdfVector, objectDatabase, nameDatabase );
              println( "Recognition Result for Object " + object + " : " + result[0] + ", minDistance: " + result[1] );
              
              counter++;
              
              int prediction = Integer.parseInt( result[0] );
              int groundTruth = object;
              
              if( groundTruth == prediction )
                numOfTrueDetections++;
              else
                numOfFalseDetections++;
            }
          }
          else
          {
            println( "Database is empty!" );
            println( "-------------------------------------------" );
          }
        }
      }
    }
  }
  
  println( "-------------------------------------------" );
  println( "Testing finished!" );
  
  accuracy = ( (float) numOfTrueDetections ) / ( counter == 0 ? 1 : counter );
  
  println( "-------------------------------------------" );
  println( "Number of Instances: " + counter );
  println( "Number of True Detections: " + numOfTrueDetections );
  println( "Number of False Detections: " + numOfFalseDetections );
  println( "Accuracy: " + accuracy );
  println( "-------------------------------------------" );
  
  long endTime = System.currentTimeMillis();
  
  println( "Elapsed Time: " + ( endTime - startTime ) + " ms." );
  println( "Average Elapsed Time for per Instance: " + ( endTime - startTime ) / ( counter == 0 ? 1 : counter ) + " ms." );
  println( "-------------------------------------------" );
}

void appendTextToFile( File f, String text )
{
  try
  {
    PrintWriter out = new PrintWriter( new BufferedWriter( new FileWriter( f, true ) ) );
    out.println( text );
    out.close();
  }
  catch( IOException e )
  {
    e.printStackTrace();
  }
}

void createFile( File f )
{
  File parentDir = f.getParentFile();
  
  try
  {
    parentDir.mkdirs(); 
    f.createNewFile();
  }
  catch( Exception e )
  {
    e.printStackTrace();
  }
}

public String [] compare(float [] currentObject, ArrayList<float[]> objectDb, ArrayList<String> nameDb)
{
  String [] result = new String[2]; 
  /*
   * Comparison through Normalized Cross Correlation distance metric
   */
  Vector<Float> distance = new Vector<Float>();
  
  for( float [] object : objectDb )
  {
    float dist = getDistance( object, currentObject, strDistanceMetric );
    
    distance.add( dist );
  }

  // select shortest distance
  int shortest = 0;
  float shortDist = (float) distance.get( 0 );
  
  for( int i = 1; i < distance.size(); i++ )
  {
    if( (float) distance.get( i ) < shortDist )
    {
      shortest = i;
      shortDist = (float) distance.get( i );
    }
  }
  
  result[0] = nameDb.get( shortest );
  result[1] = Float.toString( shortDist );
  
  return result;
}

float minExpDistance( float [] hist1, float [] hist2 )
{
  float[] average = new float[hist1.length];
  
  for (int i = 0; i < hist1.length; ++i)
  {
    average[i] += sqrt( hist1[i] * hist2[i] );
  }
  
  float minExpCoeff  =  1F;
  
  for( int i = 0; i < hist1.length; i++ )
  {
    float minVal = min( sqrt( hist1[i] * average[i] ), sqrt( hist2[i] * average[i] ) );
    
    minExpCoeff *= exp( minVal );
  }
  
  return 1F - log( minExpCoeff ); 
}

float bhattacharyyaDistance( float [] hist1, float [] hist2 )
{
  float bhattacharyyaCoeff = 0F;
  
  for( int i = 0; i < hist1.length; i++ )
  {
    bhattacharyyaCoeff += sqrt( hist1[i] * hist2[i] );
  }
  
  return 1F - bhattacharyyaCoeff;
}

float klDivergence(float[] p1, float[] p2) {

  float klDiv = 0.0;

  for (int i = 0; i < p1.length; ++i) {
    if (p1[i] == 0) { continue; }
    if (p2[i] == 0.0) { continue; }

  klDiv += p1[i] * log( p1[i] / p2[i] );
  }

  return 1F - exp( -klDiv / log( p1.length ) );
}

float jensenShannonDivergence(float[] p1, float[] p2)
{
  float[] average = new float[p1.length];
  
  for (int i = 0; i < p1.length; ++i)
  {
    average[i] += (p1[i] + p2[i]) / 2;
  }
  
  return (klDivergence(p1, average) + klDivergence(p2, average)) / 2;
}

float getDistance( float [] hist1, float [] hist2, String strDistanceMetric )
{
  if( strDistanceMetric == "minExpDistance" )
    return minExpDistance( hist1, hist2 );
  else if( strDistanceMetric == "bhattacharyyaDistance" )
    return bhattacharyyaDistance( hist1, hist2 );
  else if( strDistanceMetric == "KL_Divergence" )
    return klDivergence( hist1, hist2 );
  else if( strDistanceMetric == "jensenShannonDivergence" )
    return jensenShannonDivergence( hist1, hist2 );
    
  return 0F;
}

// Implementing Fisher–Yates shuffle
void shuffleArray( String [] ar, int seedNo )
{
  float rnd = rand( seedNo );
  
  for( int i = ar.length - 1; i > 0; i-- )
  {
    int index = floor( rnd * i );
    // Simple swap
    String a = ar[index];
    ar[index] = ar[i];
    ar[i] = a;
  }
}

float rand( int seedNo )
{
  int maxInteger = (int) pow( 2, 31 );
  
  BigInteger a = new BigInteger("1103515245");
  BigInteger c = new BigInteger("12345");
  BigInteger m = new BigInteger( String.valueOf( maxInteger ) );
  
  if( seedNo == 1 )
  {
    seed1 = ( ( a.multiply( seed1 ) ).add( c ) ).mod( m );
    return ( (float) seed1.intValue() ) / maxInteger;
  }
  else if( seedNo == 2 )
  {
    seed2 = ( ( a.multiply( seed2 ) ).add( c ) ).mod( m );
    return ( (float) seed2.intValue() ) / maxInteger;
  }
  
  return 0F;
}

PImage getDCTEnhancedImage( PImage img )
{
  PImage out = createImage( img.width, img.height, RGB );
  
  for( int j = 0; j < img.height; j += blockSize )
  {
    for( int i = 0; i < img.width; i += blockSize )
    {
      int blockHeight = min( blockSize, img.height - j );
      int blockWidth  = min( blockSize, img.width - i );
      
      double [][] block_red   = new double[blockHeight][blockWidth];
      double [][] block_green = new double[blockHeight][blockWidth];
      double [][] block_blue  = new double[blockHeight][blockWidth];
      
      for( int l = 0; l < blockHeight; l++ )
      {
        for( int k = 0; k < blockWidth; k++ )
        {
          block_red[l][k]   = (double) red( img.pixels[ (j+l) * img.width + (i+k) ] );
          block_green[l][k] = (double) green( img.pixels[ (j+l) * img.width + (i+k) ] );
          block_blue[l][k]  = (double) blue( img.pixels[ (j+l) * img.width + (i+k) ] );
        }
      }
      
      //Horizontal 1D Discrete Cosine Transform
      for( int l = 0; l < blockHeight; l++ )
      {
        double [] row_block_red   = new double[blockWidth];
        double [] row_block_green = new double[blockWidth];
        double [] row_block_blue  = new double[blockWidth];
        
        for( int k = 0; k < blockWidth; k++ )
        {
          row_block_red[k]   =  block_red[l][k];
          row_block_green[k] =  block_green[l][k];
          row_block_blue[k]  =  block_blue[l][k];
        }
        
        double [] row_dct_block_red   = new double[blockWidth];
        double [] row_dct_block_green = new double[blockWidth];
        double [] row_dct_block_blue  = new double[blockWidth];
        
        dct_transform( row_block_red, row_dct_block_red );
        dct_transform( row_block_green, row_dct_block_green );
        dct_transform( row_block_blue, row_dct_block_blue );
        
        for( int k = 0; k < blockWidth; k++ )
        {
          block_red[l][k]   = row_dct_block_red[k];
          block_green[l][k] = row_dct_block_green[k];
          block_blue[l][k]  = row_dct_block_blue[k];
        }
      }
      
      //Vertical 1D DCT
      for( int k = 0; k < blockWidth; k++ )
      {
        double [] column_block_red   = new double[blockHeight];
        double [] column_block_green = new double[blockHeight];
        double [] column_block_blue  = new double[blockHeight];
        
        for( int l = 0; l < blockHeight; l++ )
        {
          column_block_red[l]   = block_red[l][k];
          column_block_green[l] = block_green[l][k];
          column_block_blue[l]  = block_blue[l][k];
        }
        
        double [] column_dct_block_red   = new double[blockHeight];
        double [] column_dct_block_green = new double[blockHeight];
        double [] column_dct_block_blue  = new double[blockHeight];
        
        dct_transform( column_block_red, column_dct_block_red );
        dct_inverse_transform( column_dct_block_red, column_block_red );
        
        dct_transform( column_block_green, column_dct_block_green );
        dct_inverse_transform( column_dct_block_green, column_block_green );
        
        dct_transform( column_block_blue, column_dct_block_blue );
        dct_inverse_transform( column_dct_block_blue, column_block_blue );
        
        for( int l = 0; l < blockHeight; l++ )
        {
          block_red[l][k]   = column_block_red[l];
          block_green[l][k] = column_block_green[l];
          block_blue[l][k]  = column_block_blue[l];
        }
      }
      
      //Horizontal 1D Inverse Discrete Cosine Transform
      for( int l = 0; l < blockHeight; l++ )
      {
        double [] row_block_red   = new double[blockWidth];
        double [] row_block_green = new double[blockWidth];
        double [] row_block_blue  = new double[blockWidth];
        
        for( int k = 0; k < blockWidth; k++ )
        {
          row_block_red[k]   = block_red[l][k];
          row_block_green[k] = block_green[l][k];
          row_block_blue[k]  = block_blue[l][k];
        }
        
        double [] row_inverse_dct_block_red   = new double[blockWidth];
        double [] row_inverse_dct_block_green = new double[blockWidth];
        double [] row_inverse_dct_block_blue  = new double[blockWidth];
        
        dct_inverse_transform( row_block_red, row_inverse_dct_block_red );
        dct_inverse_transform( row_block_green, row_inverse_dct_block_green );
        dct_inverse_transform( row_block_blue, row_inverse_dct_block_blue );
        
        for( int k = 0; k < blockWidth; k++ )
        {
          block_red[l][k]   = row_inverse_dct_block_red[k];
          block_green[l][k] = row_inverse_dct_block_green[k];
          block_blue[l][k]  = row_inverse_dct_block_blue[k];
        }
      }
      
      for( int l = 0; l < blockHeight; l++ )
      {
        for( int k = 0; k < blockWidth; k++ )
        {
          int inputIntensity_red    =  (int) red( img.pixels[ (j+l) * img.width + (i+k) ] );
          int inputIntensity_green  =  (int) green( img.pixels[ (j+l) * img.width + (i+k) ] );
          int inputIntensity_blue   =  (int) blue( img.pixels[ (j+l) * img.width + (i+k) ] );
          
          out.pixels[ (j+l) * img.width + (i+k) ] = color( (int) Math.round( Math.sqrt( block_red[l][k] * inputIntensity_red ) ), (int) Math.round( Math.sqrt( block_green[l][k] * inputIntensity_green ) ), (int) Math.round( Math.sqrt( block_blue[l][k] * inputIntensity_blue ) ) );
        }
      }
    }
  }
  
  out.updatePixels();
  
  return out;
}

void dct_transform( double [] x, double [] X )
{
  int N = x.length;
  
  for( int k = 0; k < N; ++k )
  {
    double sum = 0D;
    double s = ( k == 0 ) ? Math.sqrt( 0.5D ) : 1D;
    
    double weight = N / ( (double) ( N + k ) );
    
    for( int n = 0; n < N; ++n )
    {
      sum += s * x[n] * Math.cos( Math.PI * ( n + 0.5D ) * ( ( (double) k ) / N ) ) * weight;
    }
    
    X[k] = sum * Math.sqrt( 2D / N );
  }
}

void dct_inverse_transform( double [] X, double [] x )
{
  int N = X.length;
  
  for( int n = 0; n < N; ++n )
  {
    double sum = 0D;
    
    for( int k = 0; k < N; ++k )
    {
      double s = ( k == 0 ) ? Math.sqrt( 0.5D ) : 1D;
      
      double weight = N / ( (double) ( N - k ) );
        
      sum += s * X[k] * Math.cos( Math.PI * ( n + 0.5D ) * ( ( (double) k ) / N ) ) * weight;
    }
    
    x[n] = sum * Math.sqrt( 2D / N );
  }
}
