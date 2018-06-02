import breeze.linalg._

//                          myu   -- myu^2 +  ...\_ KL_D
//     -- A -- S --myu      sigma -- sigma^2 +.../
// x <
//     -- A -- S --sigma
//                          z=myu+sig -- A -- T -- y

object VAE{
  val rand=new util.Random(0)
  def main(args:Array[String]){

    val mode = args(0)
    val ln   = args(1).toInt
    val dn   = args(2).toInt
    val where = args(3)
    val how = args(4)

    if(how == "learning"){
      train(mode,ln,dn,where)
    }else if(how == "test"){
      test(mode,ln,dn,where)
    }else if(how == "addtrain"){
      addtrain(mode,ln,dn,where)
    }else{
      println("miss input")
    }

    val pathName = "hist.txt"
    // run python
    scala.sys.process.Process(
      s"ipython py_program/myHist.py $pathName"
    ).run

  }

  def train(mode:String,ln:Int,dn:Int,where:String){
    val mn = new mnist()
    val (dtrain, dtest) = where match {
      case "home" =>{
        mn.load_mnist("C:/Users/poler/Documents/python/share/mnist")
      }
      case "lab" =>{
        mn.load_mnist("/home/share/number")
      }
    }
    val pre_E   = vaeNet.select_pre(mode)
    val layers  = pre_E 
    val myu_E   = vaeNet.select_myu(mode)
    val sigma_E = vaeNet.select_sigma(mode)
    val decoder = vaeNet.select_decoder(mode)

    var trainList = List[Double]()    //
    var train_inList = List[Double]() //    
    var ys = List[Array[Double]]()    //
    var xentoropyList = List[Double]()
    var MSEList = List[Double]()
    var DKLList = List[Double]()

    //start
    var zlist = List((Double,Double))
    var zs = ""
    for (i <- 0 until ln){
      var start_a = System.currentTimeMillis 
     
      //training
      var E_train = List[Double]()
      var D_train = List[Double]()
      var X_entropy = List[Double]()
     
      var num = 1
      for((x,n) <- (dtrain.toList).take(dn)){
        var err1    = 0d; var err2= 0d
        var eplist = new Array[Double](dn)
        for(i <- 0 until eplist.size){
          eplist(i) = rand.nextGaussian*0.1
        }
        val pre = forwards(layers,x)
        val ave = forwards(myu_E,pre)
        val dis = forwards(sigma_E,pre)
        val z = dis.zip(eplist).map{case (a,b) => a*b}.zip(ave).map{case (a,b) => a + b}
        
        if(ln == i+1){
          zs += n+","+ z(0).toString +","+ z(1).toString + "\n"
        }

        val y = forwards(decoder,z)
      //  y.foreach{println(_)}
        ys ::= y
       
        err1 = mse(y,x).sum
        err2 = divergence(ave,dis).sum
        E_train   ::= err1
        D_train   ::= err2
        X_entropy ::= crossEntropyCost(y,x)
        

        /*backward*/
        val d1 = backwards(decoder,x.zip(y).map{    
          case (a,b) => ((b-a)/b*(1d-b))
        } )
         
        //val d1 = backwards(decoder,mse(y,x))
       
        val d = d1.zip(eplist).map{case (a,b) => a*b}
        val aveD = d1.zip(ave).map{case (a,b) => a+b}
        val tmp = dis.map(a => a - 1/(a+0.00000001))
        val disD = d.zip(tmp).map{case (a,b) => a+b}
        val x1 = backwards(myu_E, aveD)
        val x2 = backwards(sigma_E, disD)
        val dx = x1.zip(x2).map{case (a,b) => a+b }
        backwards(layers,dx)
      
        // update
        updates(layers)
        updates(myu_E)
        updates(sigma_E)
        updates(decoder)

        if(ln-1 == i){
          trainList ::=  y.sum
          train_inList ::= x.sum
        }
      }    

      ys = ys.reverse
      if(i  == ln-1){
        Image.write("VAE/final_train_mode_"+mode+"_"+i.toString+".png",Image.make_image3(ys.toArray,10,10,28,28))
     
        print("z_make data")
        val pathName = "VAE/zdata_"+mode+".csv"
        val writer1 =  new java.io.PrintWriter(pathName)
        writer1.write(zs)
        writer1.close()

        println("----finish")

      }

      var time = System.currentTimeMillis - start_a

      //print
      //学習回数　かかった時間　誤差１…4　
      //学習時正解データ数　テスト時正解データ数 学習データ数　テストデータ数
      learning.print_result(
        i,time,List(X_entropy.sum/dn,E_train.sum/dn,D_train.sum/dn),0,0,0,0)
      
      xentoropyList ::= X_entropy.sum/dn
      MSEList ::= E_train.sum/dn
      DKLList ::= D_train.sum/dn

    }

    val pathName = "VAE/hist_train_"+mode+".txt"
    val writer =  new java.io.PrintWriter(pathName)
    val title = "train_Hist_"+mode + ".png\n"
    val ys1 = trainList.reverse.mkString(",") + "\n"
    val ys3 = train_inList.reverse.mkString(",") + "\n"
    writer.write(ys1)
    writer.write(ys3)
    writer.close()

    val pathName2 = "VAE/train_Xentoropy_"+mode+".txt"
    val writer2 =  new java.io.PrintWriter(pathName2)
    val ys2 = xentoropyList.reverse.mkString(",") + "\n"
    writer2.write(ys2)
    writer2.close()

    val pathName3 = "VAE/train_MSE_"+mode+".txt"
    val writer3 =  new java.io.PrintWriter(pathName3)
    val ys33 = xentoropyList.reverse.mkString(",") + "\n"
    writer3.write(ys33)
    writer3.close()

    val pathName4 = "VAE/train_DKL_"+mode+".txt"
    val writer4 =  new java.io.PrintWriter(pathName4)
    val ys4 = xentoropyList.reverse.mkString(",") + "\n"
    writer4.write(ys4)
    writer4.close()

  

    saves(layers,"layers_"+mode)
    saves(myu_E,"myu_"+mode)
    saves(sigma_E,"sigma_"+mode)
    saves(decoder,"decoder_"+mode)

  }


  def test(mode:String,ln:Int,dn:Int,where:String){
    val mn = new mnist()
    val (dtrain, dtest) = where match {
      case "home" =>{
        mn.load_mnist("C:/Users/poler/Documents/python/share/mnist")
      }
      case "lab" =>{
        mn.load_mnist("/home/share/number")
      }
    }
    val pre_E   = vaeNet.select_pre(mode)
    val layers  = pre_E
    val myu_E   = vaeNet.select_myu(mode)
    val sigma_E = vaeNet.select_sigma(mode)
    val decoder = vaeNet.select_decoder(mode)

    loads(layers,"layers_"+mode)
    loads(myu_E,"myu_"+mode)
    loads(sigma_E,"sigma_"+mode)
    loads(decoder,"decoder_"+mode)
    println("load finish")
  
    var test_inList = List[Double]()  //
    var testList = List[Double]()

    for(i <- 0 until ln){
      //test
      var start_t = System.currentTimeMillis
      var eError  = 0d; var dError  = 0d
      var testE = List[Double]()
      var testD = List[Double]()
      var as = List[Array[Double]]()
      for((x,n) <- (dtrain.toList).take(dn)){
        // forwards
        val pre = forwards(layers,x)
        val ave = forwards(myu_E,pre)
        val dis = forwards(sigma_E,pre)
        val z = dis.map(_*rand.nextGaussian).zip(ave).map{case (a,b) => a + b}
        val y = forwards(decoder,z)
      
        as ::= y
        // error
        eError = crossEntropyCost(y,x)
        dError = divergence(ave, dis).sum
        testE ::= eError
        testD ::= dError

        // resets
        resets(layers)
        resets(myu_E)
        resets(sigma_E)
        resets(decoder)

        if(ln-1 == i){
          testList ::= y.sum
          test_inList ::= x.sum
        }
      }

      as = as.reverse
      if(i == ln-1){
        Image.write("VAE/test_mode"+mode+"_"+i.toString+".png",Image.make_image3(as.toArray,10,10,28,28))
     }

      var Ttime =  System.currentTimeMillis -start_t 
     
      learning.print_result(
        i,Ttime,List(testE.sum/dn,testD.sum/dn),0,0,0,0)
    

    }
    val pathName = "VAE/hist_test.txt"
    val writer =  new java.io.PrintWriter(pathName)
    val ys2 = testList.reverse.mkString(",") + "\n"
    val ys3 = test_inList.reverse.mkString(",") + "\n"
    
 
    writer.write(ys2)
    writer.write(ys3)
    writer.close()


  }


  def addtrain(mode:String,ln:Int,dn:Int,where:String){
    val mn = new mnist()
    val (dtrain, dtest) = where match {
      case "home" =>{
        mn.load_mnist("C:/Users/poler/Documents/python/share/mnist")
      }
      case "lab" =>{
        mn.load_mnist("/home/share/number")
      }
    }
    val pre_E   = vaeNet.select_pre(mode)
    val layers  = pre_E 
    val myu_E   = vaeNet.select_myu(mode)
    val sigma_E = vaeNet.select_sigma(mode)
    val decoder = vaeNet.select_decoder(mode)

    var trainList = List[Double]()    //
    var train_inList = List[Double]() //    
    var ys = List[Array[Double]]()    //
    var xentoropyList = List[Double]()
    var MSEList = List[Double]()
    var DKLList = List[Double]()

    loads(layers,"layers_"+mode)
    loads(myu_E,"myu_"+mode)
    loads(sigma_E,"sigma_"+mode)
    loads(decoder,"decoder_"+mode)
    println("load finish")
  


    //start
    var zlist = List((Double,Double))
    var zs = ""
    for (i <- 0 until ln){
      var start_a = System.currentTimeMillis 
     
      //training
      var E_train = List[Double]()
      var D_train = List[Double]()
      var X_entropy = List[Double]()
     
      var num = 1
      for((x,n) <- (dtrain.toList).take(dn)){
        var err1    = 0d; var err2= 0d
        var eplist = new Array[Double](dn)
        for(i <- 0 until eplist.size){
          eplist(i) = rand.nextGaussian*0.1
        }
        val pre = forwards(layers,x)
        val ave = forwards(myu_E,pre)
        val dis = forwards(sigma_E,pre)
        val z = dis.zip(eplist).map{case (a,b) => a*b}.zip(ave).map{case (a,b) => a + b}
        
    
        val y = forwards(decoder,z)
       
        err1 = mse(y,x).sum
        err2 = divergence(ave,dis).sum
        E_train   ::= err1
        D_train   ::= err2
        X_entropy ::= crossEntropyCost(y,x)
        

        /*backward*/
        val d1 = backwards(decoder,x.zip(y).map{    
          case (a,b) => ((b-a)/b*(1d-b))
        } )
         
        //val d1 = backwards(decoder,mse(y,x))
       
        val d = d1.zip(eplist).map{case (a,b) => a*b}
        val aveD = d1.zip(ave).map{case (a,b) => a+b}
        val tmp = dis.map(a => a - 1/(a+0.00000001))
        val disD = d.zip(tmp).map{case (a,b) => a+b}
        val x1 = backwards(myu_E, aveD)
        val x2 = backwards(sigma_E, disD)
        val dx = x1.zip(x2).map{case (a,b) => a+b }
        backwards(layers,dx)
      
        // update
        updates(layers)
        updates(myu_E)
        updates(sigma_E)
        updates(decoder)

      }

    


      var time = System.currentTimeMillis - start_a

      //print
      //学習回数　かかった時間　誤差１…4　
    //学習時正解データ数　テスト時正解データ数 学習データ数　テストデータ数
      learning.print_result(
        i,time,List(X_entropy.sum/dn,E_train.sum/dn,D_train.sum/dn),0,0,0,0)

      xentoropyList ::= X_entropy.sum/dn
      MSEList ::= E_train.sum/dn
      DKLList ::= D_train.sum/dn

    }


    saves(layers,"layers_"+mode)
    saves(myu_E,"myu_"+mode)
    saves(sigma_E,"sigma_"+mode)
    saves(decoder,"decoder_"+mode)

  }





  def crossEntropyCost(y:Array[Double],x:Array[Double])={
    var sum = 0d
    if(y.size != x.size){println("size err")}
    for(i <- 0 until x.size){
      sum -= x(i) * math.log(1e-08 + y(i)) + (1 - x(i)) * math.log(1e-08 + 1 - y(i))
    }
    
    sum
  }

  def divergence(ave:Array[Double],dis:Array[Double])={
    var returnfile = new Array[Double](ave.size)
    for(i <- 0 until ave.size){
      returnfile(i)=(ave(i)*ave(i) + dis(i)*dis(i) -1 -math.log(dis(i)*dis(i)))/2
    }

    returnfile
  }
  def mse(x:Array[Double],y:Array[Double])={
    var returnfile = new Array[Double](x.size)

    for(i <- 0 until x.size){
      returnfile(i) = y(i)-x(i)
    }

    returnfile
  }

  def createInput(size:Int)={
    val nums =
      for(i <- 0 until size) yield
        Array(rand.nextGaussian*0.1)
    nums.toArray
  }

  def forwards(layers:List[Layer],x:Array[Double])={
    var temp = x
    for(lay <- layers){temp =lay.forward(temp) }
    temp
  }

  def backwards(layers:List[Layer],x:Array[Double])={
    var d = x
    for(lay <- layers.reverse){d = lay.backward(d)}
    d
  }

  def updates(layers:List[Layer])={
    for(lay <- layers){lay.update()}
  }

  def resets(layers:List[Layer]){
    for(lay <- layers){lay.reset()}
  }

  def saves(layers:List[Layer],fn:String){
    
    for(i <- 0 until layers.size){
      layers(i).save("biasdata/"+fn+"_"+i.toString)
    }
  }
  def loads(layers:List[Layer],fn:String){
    for(i <- 0 until layers.size){
      layers(i).load("biasdata/"+fn+"_"+i.toString)
    }
  }
}



