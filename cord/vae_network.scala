import breeze.linalg._

object vaeNet{
  def select_pre(mode:String)={
    val pre_E = mode match{
      case "0" =>{
        val a0 = new Affine(28*28,100)
        val b0 = new Affine(100,2)

        List(a0,b0)
      }
      case "z20" =>{
        val z201 = new Affine(28*28,100)
        val z20_t1 = new Tanh()
        val z202 = new Affine(100,100)
        val z20_t2 = new Tanh()
        val z203 = new Affine(100,20)

        List(z201,z20_t1,z202,z20_t2,z203)
      }
      case "z2" =>{
        val z21 = new Affine(28*28,100)
        val b2 = new Tanh()
        val z22 = new Affine(100,100)
        val b4 = new Tanh()
        val z23 = new Affine(100,2)

        List(z21,b2,z22,b4,z23)
      }

      case "z20_1" =>{
        val z201 = new Affine(28*28,100)
        val z20_t1 = new Sigmoid()
        val z202 = new Affine(100,100)
        val z20_t2 = new Sigmoid()
        val z203 = new Affine(100,20)

        List(z201,z20_t1,z202,z20_t2,z203)
      }
      case "z2_1" =>{
        val z21 = new Affine(28*28,100)
        val b2 = new Sigmoid()
        val z22 = new Affine(100,100)
        val b4 = new Sigmoid()
        val z23 = new Affine(100,2)

        List(z21,b2,z22,b4,z23)
      }

      case "z20_2" =>{
        val z201 = new Affine(28*28,100)
        val z20_t1 = new ReLU()
        val z202 = new Affine(100,100)
        val z20_t2 = new ReLU()
        val z203 = new Affine(100,20)

        List(z201,z20_t1,z202,z20_t2,z203)
      }
      case "z2_2" =>{
        val z21 = new Affine(28*28,100)
        val b2 = new ReLU()
        val z22 = new Affine(100,100)
        val b4 = new ReLU()
        val z23 = new Affine(100,2)

        List(z21,b2,z22,b4,z23)
      }
    }

    pre_E
  }



  def select_myu(mode:String) ={
    val myu_E =mode match{
      case "z2" =>{
        val ae1 = new Affine(2,2)
        val be1 = new Sigmoid()
        List(ae1,be1)
      }

      case "z20" =>{
        val ae2 = new Affine(20,20)
        val be2 = new Sigmoid()
        List(ae2,be2)
      }

      case "z2_1" =>{
        val ae1 = new Affine(2,2)
        val be1 = new Sigmoid()
        List(ae1,be1)
      }

      case "z20_1" =>{
        val ae2 = new Affine(20,20)
        val be2 = new Sigmoid()
        List(ae2,be2)
      }

      case "z2_2" =>{
        val ae1 = new Affine(2,2)
        val be1 = new Sigmoid()
        List(ae1,be1)
      }

      case "z20_2" =>{
        val ae2 = new Affine(20,20)
        val be2 = new Sigmoid()
        List(ae2,be2)
      }

      case "min" =>{
        val min1 = new Affine(28*28,2)
        val min2 = new Sigmoid()
        List(min1,min2)
      }
    }

    myu_E
  }
  def select_sigma(mode:String)={
    val sigma_E = mode match{
      case "z2" => {
        val as1 = new Affine(2,2)
        val bs1 = new Sigmoid()
        List(as1,bs1)
      }

      case "z20" => {
        val as2 = new Affine(20,20)
        val bs2 = new Sigmoid()
        List(as2,bs2)
      }
      case "z2_1" => {
        val as1 = new Affine(2,2)
        val bs1 = new Sigmoid()
        List(as1,bs1)
      }

      case "z20_1" => {
        val as2 = new Affine(20,20)
        val bs2 = new Sigmoid()
        List(as2,bs2)
      }
      case "z2_2" => {
        val as1 = new Affine(2,2)
        val bs1 = new Sigmoid()
        List(as1,bs1)
      }

      case "z20_2" => {
        val as2 = new Affine(20,20)
        val bs2 = new Sigmoid()
        List(as2,bs2)
      }

      case "min" => {
        val min1 = new Affine (28*28,2)
        val min2 = new Sigmoid()
        List(min1,min2)
      }

    }
    sigma_E
  }
 
  def select_decoder(mode:String)={
    val decoder = mode match{
      case "0" => {
        val a2 = new Affine(2,28*28)
        val b2 = new Sigmoid()
        List(a2,b2)
      }
      case "z20" => {
        val a1 = new Affine(20,100)
        val a2 = new Tanh()
        val a3 = new Affine(100,100)
        val a4 = new Tanh()
        val a5 = new Affine(100,100)
        val a6 = new Tanh()
        val a7 = new Affine(100,28*28)

        val b2 = new Sigmoid()
        List(a1,a2,a3,a4,a5,a6,a7,b2)
      }


      case "z2" => {
        val a1 = new Affine(2,100)
        val a2 = new Tanh()
        val a3 = new Affine(100,100)
        val a4 = new Tanh()
        val a5 = new Affine(100,100)
        val a6 = new Tanh()
        val a7 = new Affine(100,28*28)

        val b2 = new Sigmoid()
        List(a1,a2,a3,a4,a5,a6,a7,b2)
      }

      case "z20_1" => {
        val a1 = new Affine(20,100)
        val a2 = new Sigmoid()
        val a3 = new Affine(100,100)
        val a4 = new Sigmoid()
        val a5 = new Affine(100,100)
        val a6 = new Sigmoid()
        val a7 = new Affine(100,28*28)

        val b2 = new Sigmoid()
        List(a1,a2,a3,a4,a5,a6,a7,b2)
      }


      case "z2_1" => {
        val a1 = new Affine(2,100)
        val a2 = new Sigmoid()
        val a3 = new Affine(100,100)
        val a4 = new Sigmoid()
        val a5 = new Affine(100,100)
        val a6 = new Sigmoid()
        val a7 = new Affine(100,28*28)

        val b2 = new Sigmoid()
        List(a1,a2,a3,a4,a5,a6,a7,b2)
      }

      case "z20_2" => {
        val a1 = new Affine(20,100)
        val a2 = new ReLU()
        val a3 = new Affine(100,100)
        val a4 = new ReLU()
        val a5 = new Affine(100,100)
        val a6 = new ReLU()
        val a7 = new Affine(100,28*28)

        val b2 = new Sigmoid()
        List(a1,a2,a3,a4,a5,a6,a7,b2)
      }


      case "z2_2" => {
        val a1 = new Affine(2,100)
        val a2 = new ReLU()
        val a3 = new Affine(100,100)
        val a4 = new ReLU()
        val a5 = new Affine(100,100)
        val a6 = new ReLU()
        val a7 = new Affine(100,28*28)

        val b2 = new Sigmoid()
        List(a1,a2,a3,a4,a5,a6,a7,b2)
      }


      case "min" =>{
        val min1 = new Affine(2,28*28)
        val min2 = new Sigmoid()
        List(min1,min2)
      }
    }
    decoder
  }


  

}

