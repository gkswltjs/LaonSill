/*
 * UbyteImage.h
 *
 *  Created on: 2016. 7. 14.
 *      Author: jhkim
 */

#ifndef UBYTEIMAGE_H_
#define UBYTEIMAGE_H_

#define UBYTE_IMAGE_MAGIC 2051
#define UBYTE_LABEL_MAGIC 2049

#ifdef _MSC_VER
	#define bswap(x) _byteswap_ulong(x)
#else
	#define bswap(x) __builtin_bswap32(x)
#endif


struct UByteImageDataset {
	uint32_t magic;			/// Magic number (UBYTE_IMAGE_MAGIC).
	uint32_t length;		/// Number of images in dataset.
	uint32_t height;		/// The height of each image.
	uint32_t width;			/// The width of each image.
	uint32_t channel;		/// The channel of each image.
	void Swap() {
		magic = bswap(magic);
		length = bswap(length);
		height = bswap(height);
		width = bswap(width);
		channel = bswap(channel);
	}
};

struct UByteLabelDataset {
	uint32_t magic;			/// Magic number (UBYTE_LABEL_MAGIC).
	uint32_t length;		/// Number of labels in dataset.
	void Swap() {
		magic = bswap(magic);
		length = bswap(length);
	}
};




#endif /* UBYTEIMAGE_H_ */
