#ifdef _WIN32
	#define DLL_EXPORT __declspec(dllexport)
#else
	#define DLL_EXPORT
#endif

#include <stdio.h>
#include <jpeglib.h>

short setParity(short c, int bit) {
	bit &= 1;
	if (c == 0) return (bit == 0) ? 0 : 1;
	if ((c & 1) == bit) return c;
	return (short)(c + (c > 0 ? 1 : -1));
}

DLL_EXPORT int embedding(
	const char* jpgInPath, const char* jpgOutPath,
	const unsigned char* bits, size_t n_bits,
	const unsigned char* usable,
	int marginX, int marginY,
	const int (*uvList)[2], int uvCount,
	int qrSize, int allumerF, int allumerS,
	int bbox_x1, int bbox_y1,
	int bbox_x2, int bbox_y2
	) {

	//printf("Start Working\n");

	FILE* inputImage = fopen(jpgInPath, "rb");
	FILE* outputImage = fopen(jpgOutPath, "wb");
	if (!inputImage || !outputImage) {
		printf("File Error");
		return 0;
	}

	//printf("File Load Complete\n");

	struct jpeg_decompress_struct src;
	struct jpeg_compress_struct dst;
	struct jpeg_error_mgr jerr_src, jerr_dst;

	src.err = jpeg_std_error(&jerr_src);
	jpeg_create_decompress(&src);
	jpeg_stdio_src(&src, inputImage);
	if (jpeg_read_header(&src, TRUE) != JPEG_HEADER_OK) {
		printf("No Header\n");
		return 0;
	}

	jvirt_barray_ptr* coefArrays = jpeg_read_coefficients(&src);
	if (!coefArrays) {
		printf("No COEF\n");
		return 0;
	}

	jpeg_component_info* ci = &src.comp_info[0];
	int blockW = (int)ci->width_in_blocks;
	int blockH = (int)ci->height_in_blocks;	// 블록 개수

	//printf("Insert Stegano\n");

	int bx1 = bbox_x1 / 8;
	int bx2 = (bbox_x2 - 1) / 8;
	int by1 = bbox_y1 / 8;
	int by2 = (bbox_y2 - 1) / 8;

	int frontiereUV[2][2] = { {1, 2}, {2, 1} };

	for (int by = by1; by <= by2; by++) {
		JBLOCKARRAY row = (JBLOCKARRAY)(*src.mem->access_virt_barray)(
			(j_common_ptr)&src, coefArrays[0], (JDIMENSION)by, (JDIMENSION)1, TRUE);

		for (int bx = bx1; bx <= bx2; bx++) {
			if(bx == bx1 || bx == bx2 || by == by1 || by == by2){
				JBLOCK* blk = &row[0][bx];

				for (int k = 0; k < 2; k++) {
					int pos = frontiereUV[k][0] * 8 + frontiereUV[k][1];
					(*blk)[pos] = (JCOEF)setParity((short)(*blk)[pos], 1);
				}

				if (allumerF) {
					(*blk)[1] = (JCOEF)(200);
				}
			}
		}
	}

	int dcAdd = (allumerS == 1) ? 200 : 0;

	size_t usableCount = 0;
	int rowCount = 0;
	for (int by = 0; by < qrSize && usableCount < n_bits; by++) {
		for (int bx = marginX; bx < blockW && usableCount < n_bits; bx++) {
			int blockNum = ((rowCount + marginY) * blockW) + bx;
			if (!usable[blockNum]) continue;

			JBLOCKARRAY row = (JBLOCKARRAY)(*src.mem->access_virt_barray)(
				(j_common_ptr)&src, coefArrays[0], (JDIMENSION)(by + marginY), (JDIMENSION)1, TRUE);
			JBLOCK* blk = &row[0][bx];

			(*blk)[0] = (JCOEF)((int)(*blk)[0] + dcAdd);

			int bit = bits[usableCount];
			for (int k = 0; k < uvCount; k++) {
				int u = uvList[k][0], v = uvList[k][1];
				int pos = u * 8 + v;
				(*blk)[pos] = (JCOEF)setParity((short)(*blk)[pos], bit);
			}

			usableCount++;

			if (usableCount % qrSize == 0) {
				rowCount++;
				break;
			}
		}
	}

	//printf("Saving New File\n");

	dst.err = jpeg_std_error(&jerr_dst);
	jpeg_create_compress(&dst);
	jpeg_stdio_dest(&dst, outputImage);

	jpeg_copy_critical_parameters(&src, &dst);
	dst.optimize_coding = TRUE;

	jpeg_write_coefficients(&dst, coefArrays);
	jpeg_finish_compress(&dst);
	jpeg_finish_decompress(&src);

	jpeg_destroy_compress(&dst);
	jpeg_destroy_decompress(&src);

	fclose(inputImage);
	fclose(outputImage);

	//printf("Process Finished\n");
	return usableCount;
}

DLL_EXPORT int functionTest(char letter) {
	printf("%c Working!", letter);
	return 1;
}
