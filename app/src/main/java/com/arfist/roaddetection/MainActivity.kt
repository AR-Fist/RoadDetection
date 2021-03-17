package com.arfist.roaddetection

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.Rect
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.math.atan2


class MainActivity : AppCompatActivity() {

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // init OpenCV
        if(OpenCVLoader.initDebug()) {
            Log.i("OpenCV", "Load successfully")
        }
        else {
            Log.e("OpenCV", "Load fail")
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
                baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onStart() {
        super.onStart()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    @SuppressLint("UnsafeExperimentalUsageError")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Analysis
            val imageAnalysis = ImageAnalysis.Builder()
                    .build()

            imageAnalysis.setAnalyzer(cameraExecutor, { imageproxy ->
                val rotationDegrees = imageproxy.imageInfo.rotationDegrees

                // get image from image proxy
                val img = imageproxy.image?.toBitmap()
                if (img != null) {
                    Log.d("Image", img.height.toString() + ", " + img.width.toString())
                }

                var img_canny = img?.let { detectEdges(it) }
                var mat_roi = img_canny?.let { roi(it) }

                var lines = Mat()
                var mat_roi_gray = Mat()
                Imgproc.cvtColor(mat_roi, mat_roi_gray, Imgproc.COLOR_RGB2GRAY, 4)
                Imgproc.HoughLinesP(mat_roi_gray, lines, 1.0, PI / 180.0, 50, 30.0, 10.0)

//                var resultMat = Mat()
                var resultMat = mat_roi
//                Utils.bitmapToMat(img, resultMat);

                Log.d("Houghlines",lines.rows().toString())

                var result_lines = img?.let { average_slope_intercept(it,lines) }

                // plot left
                var left = result_lines?.get(0)
                Log.d("Left line (x1,y1,x2,y2)",left.toString())
                var pt1 = left?.get(0)?.let { Point(it.toDouble(), left[1].toDouble()) }
                var pt2 = left?.get(2)?.let { Point(it.toDouble(), left[3].toDouble()) }
                //Drawing lines on an image
                Imgproc.line(resultMat, pt1, pt2, Scalar(255.0, 0.0, 0.0), 2)
                //Drawing dots
                Imgproc.circle(resultMat, img?.width?.let { Point(it.toDouble(),0.0) },5,Scalar(0.0,255.0,0.0),5)
                Imgproc.circle(resultMat, img?.width?.let { Point(it.toDouble(),img.height.toDouble()) },5,Scalar(0.0,255.0,0.0),5)
                Imgproc.circle(resultMat, img?.width?.let { Point(it.toDouble()/2-20,img.height.toDouble()/2-150) },5,Scalar(255.0,0.0,255.0),5)
                Imgproc.circle(resultMat, img?.width?.let { Point(it.toDouble()/2-20,img.height.toDouble()/2+150) },5,Scalar(255.0,0.0,255.0),5)

                // plot right
                var right = result_lines?.get(1)
                Log.d("Right line (x1,y1,x2,y2)",right.toString())
                pt1 = right?.get(0)?.let { Point(it.toDouble(), right[1].toDouble()) }
                pt2 = right?.get(2)?.let { Point(it.toDouble(), right[3].toDouble()) }
                //Drawing lines on an image
                Imgproc.line(resultMat, pt1, pt2, Scalar(255.0, 0.0, 0.0), 2)


                val resultBitmap = img?.copy(Bitmap.Config.RGB_565, true)
                Utils.matToBitmap(resultMat, resultBitmap)


                // UI thread for update ImageView
                runOnUiThread {
                    var imgview = findViewById<ImageView>(R.id.image_view)
                    imgview.setImageBitmap(resultBitmap)
                }
                imageproxy.close()
            })


            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis)

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    // util method for converting Image to Bitmap
    fun Image.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer // Y
        val vuBuffer = planes[2].buffer // VU

        val ySize = yBuffer.remaining()
        val vuSize = vuBuffer.remaining()

        val nv21 = ByteArray(ySize + vuSize)

        yBuffer.get(nv21, 0, ySize)
        vuBuffer.get(nv21, ySize, vuSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    // Perform edge detection
    private fun detectEdges(bitmap: Bitmap): Bitmap? {
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val gauss = Mat()
        val edges = Mat(rgba.size(), CvType.CV_8UC1)
        Imgproc.cvtColor(rgba, gauss, Imgproc.COLOR_RGB2GRAY, 4)
        Imgproc.GaussianBlur(gauss, edges, Size(3.0, 3.0), 3.0, 3.0)
        Imgproc.Canny(edges, edges, 80.0, 100.0)
        val resultBitmap = Bitmap.createBitmap(edges.cols(), edges.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(edges, resultBitmap)
        return resultBitmap
    }

    private fun roi(bitmap: Bitmap): Mat? {
        val h = bitmap.height
        val w = bitmap.width
        var img = Mat()
        Utils.bitmapToMat(bitmap, img)
        Log.d("ROI", "img >> "+ img.width().toString() + ", " + img.height().toString() + ", " + img.type().toString())
        var mask = Mat(img.height(), img.width(), CvType.CV_8UC4, Scalar(0.0, 0.0, 0.0))
        Log.d("ROI", "mask >> "+ mask.width().toString() + ", " + mask.height().toString() + ", " + mask.type().toString())
        var points: List<Point> = listOf<Point>(Point(w.toDouble(), 0.0), Point(w.toDouble(), h.toDouble()), Point(w / 2.0 - 20.0, (h / 2.0) + 150.0), Point(w / 2.0 - 20.0, (h / 2.0) - 150.0))
        var mpoints = MatOfPoint()
        mpoints.fromList(points)
        var finalpoints = ArrayList<MatOfPoint>()
        finalpoints.add(mpoints)
        Imgproc.fillPoly(mask, finalpoints, Scalar(255.0, 255.0, 255.0))
        var dst = Mat()
        Core.bitwise_and(img, mask, dst)
        Log.d("ROI", "dst >> "+ dst.width().toString() + ", " + dst.height().toString() + ", " + dst.type().toString())
        return dst
    }

    private fun average_slope_intercept(bitmap : Bitmap, lines: Mat): ArrayList<ArrayList<Int>> {
        var left_slope_mean = 0.0
        var right_slope_mean = 0.0
        var left_intercept_mean = 0.0
        var right_intercept_mean = 0.0
        var left_n = 0
        var right_n = 0
        for(i in 0 until lines.rows()) {
            val points = lines[i, 0]
            val x1 = points[0]
            val y1 = points[1]
            val x2 = points[2]
            val y2 = points[3]

            // cut out some lines
            val Angle: Double = atan2(y2 - y1, x2 - x1) * 180.0 / PI
            if( ( -90.0 <= Angle && Angle <= -60.0) || (Angle in 60.0..90.0) || ( -10.0 <= Angle && Angle <= 10.0 ) ) {
                continue
            }
            Log.d("Angle", Angle.toString())

            val slope = (y2-y1)/(x2-x1)
            val intercept = y1 - (slope * x1)
            Log.d("average_slope_intercept","slope: " + slope.toString() + ", intercept: " + intercept.toString())
            if(slope < 0) {
                left_slope_mean += slope
                left_intercept_mean += intercept
                left_n +=1
            }
            else {
                right_slope_mean += slope
                right_intercept_mean += intercept
                right_n +=1
            }
        }
        left_slope_mean /= left_n
        left_intercept_mean /= left_n
        right_slope_mean /= right_n
        right_intercept_mean /= right_n
        val leftLine = make_coor(bitmap,left_slope_mean,left_intercept_mean,"Left")
        val rightLine = make_coor(bitmap,right_slope_mean,right_intercept_mean,"Right")
        return arrayListOf(leftLine,rightLine)
    }

    private fun make_coor(bitmap: Bitmap, slope: Double, intercept: Double, side: String): ArrayList<Int> {
        val h = bitmap.height
        val w = bitmap.width
        var roadLine = arrayListOf(0, 0, 0, 0)

        // Find intercept in trapezoid

        val top_trapezoid_intersection = lineIntersection(slope,intercept, arrayListOf(w/2-20,h/2-150,w/2-20,h/2+150))
        val bottom_trapezoid_intersection = lineIntersection(slope,intercept, arrayListOf(w,0,w,h))
        val left_trapezoid_intersection = lineIntersection(slope,intercept, arrayListOf(w,0,w/2-20,h/2-150))
        val right_trapezoid_intersection = lineIntersection(slope,intercept, arrayListOf(w/2-20,h/2+150,w,h))

        if(top_trapezoid_intersection != Pair(-1,-1)) {
            if(side == "Left") {
                roadLine[0] = top_trapezoid_intersection.first
                roadLine[1] = top_trapezoid_intersection.second
            }
            else {
                roadLine[2] = top_trapezoid_intersection.first
                roadLine[3] = top_trapezoid_intersection.second
            }
        }
        if(bottom_trapezoid_intersection != Pair(-1,-1)) {
            if(side == "Left") {
                roadLine[2] = bottom_trapezoid_intersection.first
                roadLine[3] = bottom_trapezoid_intersection.second
            }
            else {
                roadLine[0] = bottom_trapezoid_intersection.first
                roadLine[1] = bottom_trapezoid_intersection.second
            }

        }
        return roadLine
    }

    private fun lineIntersection(line1_slope : Double, line1_intercept : Double , line2 : ArrayList<Int> ): Pair<Int, Int> {

        // line1 from road line, line2 from trapezoid

        val m2 = (line2[3]-line2[1])/(line2[2]-line2[0]+0.0001) // slope (divided by zero prevention)
        val b2 = line2[1] - (m2 * line2[0]) // y-intercept
        if(line1_slope == m2) {
            return Pair(-1,-1) // parallel line
        }
        else {
            val xi = (line1_intercept-b2) / (m2-line1_slope)
            val yi = line1_slope * xi + line1_intercept
            return Pair(xi.toInt(), yi.toInt())
        }
    }

    companion object {
        private const val TAG = "CameraXBasic"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}