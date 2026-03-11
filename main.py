import textwrap
from models.text_model import TextModel
from models.image_model import ImageModel


def main():
    print("\n==== AI Text & Image Generator ====\n")
    print("1. Text Generation")
    print("2. Image Generation")
    print("3. Exit\n")

    choice = input("Select option (1 / 2 / 3): ")

    if choice == "1":
        text_ai = TextModel()
        prompt = input("\nEnter text prompt: ")
        response = text_ai.generate(prompt)

        print("\nAI Response:\n")
        wrapped_text = textwrap.fill(response, width=80)
        print(wrapped_text)

    elif choice == "2":
        image_ai = ImageModel()
        prompt = input("\nEnter image prompt: ")
        path = image_ai.generate(prompt)

        print("\nImage saved at:", path)

    elif choice == "3":
        print("Exiting program...")

    else:
        print("Invalid option")


if __name__ == "__main__":
    main()